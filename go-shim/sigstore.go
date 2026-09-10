// sigstore.go – cosign-format detached signatures over model manifests,
// used by both the docker and podman backends. No build tag.
//
// llmman content-addresses and re-hashes every blob it pulls, which
// proves the bytes match what the registry said they would be and
// nothing about who put them there. That gap matters here because
// llmman has no gatekeeper: `llmman pull gemma4` resolves through
// shortnames.conf to some registry the user never typed, and the GGUF it
// lands on goes straight into llama-server's parser.
//
// # Format
//
// cosign's, so `cosign verify --key` reads what `llmman push --sign-key`
// writes. For digest sha256:<hex> in repository <repo>, the signature is
// at <repo>:sha256-<hex>.sig, an OCI image manifest with one layer per
// signature:
//
//	layer.mediaType  application/vnd.dev.cosign.simplesigning.v1+json
//	layer blob       the simple-signing payload (see simpleSigningPayload)
//	layer.annotations["dev.cosignproject.cosign/signature"]
//	                 base64 of the raw signature over those payload bytes
//
// Verification succeeds if any layer checks out against any trusted key,
// which is what makes key rotation possible.
//
// # Scope
//
// Key-based verification only. Keyless (Fulcio certificates, Rekor
// inclusion proofs, CT SCTs) is a much larger trust-root surface and is
// unusable air-gapped, which is llmman's stated audience. The format
// leaves room for it: a keyless signature is the same layer shape with
// two extra annotations.
//
// # Backend-agnostic
//
// Both backends expose pullToLayout/pushToRegistry identically and share
// every blob helper, so a signature — a very small OCI artifact — needs
// no registry code of its own: stage it in a throwaway layout and reuse
// those. The exceptions are fetchManifestRaw and resolveManifestDigest,
// which have no layout-shaped equivalent; see backend_docker.go.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"bytes"
	"context"
	"crypto"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/sigstore/sigstore/pkg/signature"
)

// cosignSignatureMediaType is the media type of a simple-signing payload
// layer inside a signature manifest.
const cosignSignatureMediaType = "application/vnd.dev.cosign.simplesigning.v1+json"

// cosignSignatureAnnotation holds the base64-encoded raw signature over
// its own layer's payload bytes.
const cosignSignatureAnnotation = "dev.cosignproject.cosign/signature"

// cosignSignatureType is the fixed `critical.type` discriminator of a
// simple-signing payload produced by cosign. Rejecting any other value
// is what stops a payload of some *other* signed-JSON scheme, which
// happened to be signed by a key we trust for models, from being
// replayed here as if it were a model signature.
const cosignSignatureType = "cosign container image signature"

// signatureTagSuffix is appended to the digest-derived tag a signature
// is published under. cosign's `.sig`; `.att` (attestations) and `.sbom`
// are the same convention with different suffixes, and are not read here.
const signatureTagSuffix = ".sig"

// sigstoreBundleArtifactTypePrefix identifies the *other* thing that can
// be sitting next to a manifest: a sigstore bundle
// (application/vnd.dev.sigstore.bundle.v0.3+json and its earlier
// versions), published at the suffix-less fallback tag
// <repo>:sha256-<hex> as an image index rather than at the `.sig` tag as
// a simple-signing manifest.
//
// llmman does not verify bundles — see detectBundleSignature for what it
// does with one instead, and why merely *recognizing* the format is
// worth the round trip it costs.
const sigstoreBundleArtifactTypePrefix = "application/vnd.dev.sigstore.bundle"

// Limits enforced from the manifest before any blob is fetched. "The
// artifact is small" is the claim of a party not yet trusted, not a
// guarantee: without these, a hostile registry could publish gigabytes
// at the .sig tag and turn the cheap pre-download check into the
// expensive download it exists to avoid.
const (
	maxManifestBytes          = 1 << 20 // 1 MiB
	maxSignatureBytes         = 1 << 16 // 64 KiB per layer
	maxSignatureArtifactBytes = 1 << 22 // 4 MiB for config + every layer
	maxSignatureCount         = 64
)

// simpleSigningPayload is the JSON document that gets signed — the
// "simple signing" format from containers-signature.5.md, which cosign
// reuses verbatim.
//
// Declared here rather than imported from sigstore's pkg/signature/
// payload: that package's type is identical but drags in
// go-containerregistry for a field on a different struct. The wire
// format is fixed by spec, so there is nothing to drift against.
//
// Field order matters for what we emit (the signature covers the exact
// bytes), not for what we read.
type simpleSigningPayload struct {
	Critical struct {
		Identity struct {
			// DockerReference is what the signer claimed to be
			// signing. Checking it is not optional — see
			// identityMatchesRepository.
			DockerReference string `json:"docker-reference"`
		} `json:"identity"`
		Image struct {
			DockerManifestDigest string `json:"docker-manifest-digest"`
		} `json:"image"`
		Type string `json:"type"`
	} `json:"critical"`
	// Optional is emitted even when nil (as `"optional":null`), matching
	// cosign byte-for-byte so a payload written here is indistinguishable
	// from one cosign would have written for the same inputs.
	Optional map[string]any `json:"optional"`
}

// ---------------------------------------------------------------------------
// Reference arithmetic
// ---------------------------------------------------------------------------

// repositoryOf strips any tag or digest from ref, leaving
// host/namespace/name. A ":" only starts a tag if it comes after the
// last "/", so a registry port ("host:5000/owner/repo") survives — the
// same rule normalizeTag (classify.go) applies.
func repositoryOf(ref string) string {
	if i := strings.LastIndex(ref, "@"); i >= 0 {
		ref = ref[:i]
	}
	if i := strings.LastIndex(ref, ":"); i > strings.LastIndex(ref, "/") {
		ref = ref[:i]
	}
	return ref
}

// canonicalRepository folds the spellings of one Docker Hub repository
// onto a single fully-qualified form, so identityMatchesRepository can
// compare them: llmman's bare-name default resolves to
// docker.io/ai/<name> while cosign writes index.docker.io/ai/<name>.
//
// Normalizes *up*, keeping the host. Stripping it made
// "docker.io/quay.io/org/m" collapse onto the genuinely different
// "quay.io/org/m" — a transplant path. Mirrors src/verify.rs.
func canonicalRepository(repo string) string {
	for _, alias := range []string{"index.docker.io", "registry-1.docker.io"} {
		if repo == alias {
			return "docker.io"
		}
		if rest, ok := strings.CutPrefix(repo, alias+"/"); ok {
			return canonicalRepository("docker.io/" + rest)
		}
	}
	first, rest, hasSlash := strings.Cut(repo, "/")
	// Docker's own rule for telling a registry host from a namespace.
	if !strings.ContainsAny(first, ".:") && first != "localhost" {
		if hasSlash {
			return "docker.io/" + first + "/" + rest
		}
		return "docker.io/library/" + first
	}
	// An official image: "docker.io/ubuntu" is "docker.io/library/ubuntu".
	if path, ok := strings.CutPrefix(repo, "docker.io/"); ok && !strings.Contains(path, "/") {
		return "docker.io/library/" + path
	}
	return repo
}

// signatureTagFor returns the reference a signature over target is
// published at: the same repository as ref, tagged sha256-<hex>.sig.
func signatureTagFor(ref string, target digest.Digest) string {
	tag := strings.Replace(target.String(), ":", "-", 1) + signatureTagSuffix
	return repositoryOf(ref) + ":" + tag
}

// identityMatchesRepository reports whether a payload's claimed
// docker-reference names the repository actually pulled from.
//
// Without this the scheme is defeated by copying: a signature binds to a
// digest, not a location, so an attacker can copy a legitimately signed
// model *and* its signature into their own repository and have both
// verify against the real publisher's key. Sigstore's own payload docs
// say the same ("ALMOST ALL consumers MUST verify that ClaimedIdentity
// in the signature is correct given how user refers to the image").
//
// Repositories, not full references: one digest is legitimately
// reachable under many tags, and the digest binding makes that safe —
// a tag can only move to some other digest, which has its own signature.
// An empty claim is rejected; it asserts nothing about where it belongs.
func identityMatchesRepository(claimed, ref string) bool {
	if strings.TrimSpace(claimed) == "" {
		return false
	}
	return canonicalRepository(repositoryOf(claimed)) == canonicalRepository(repositoryOf(ref))
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

// trustedKey pairs a loaded verifier with the path it came from, so a
// successful match can report *which* key accepted the signature rather
// than just that one did.
type trustedKey struct {
	path     string
	verifier signature.Verifier
}

// loadTrustedKeys reads each PEM public-key file into a verifier.
// SHA-256 is ignored for ed25519 (the key type picks the algorithm) and
// is what cosign uses for ECDSA and RSA.
//
// An unloadable key is a hard error, not a skipped entry: quietly
// continuing would narrow the trust set — to nothing, in the single-key
// case, which then reads as "no trusted key accepted this signature"
// rather than "your key is unreadable".
func loadTrustedKeys(paths []string) ([]trustedKey, error) {
	keys := make([]trustedKey, 0, len(paths))
	for _, path := range paths {
		v, err := signature.LoadVerifierFromPEMFile(path, crypto.SHA256)
		if err != nil {
			return nil, fmt.Errorf("load public key %s: %w", path, err)
		}
		keys = append(keys, trustedKey{path: path, verifier: v})
	}
	return keys, nil
}

// signatureMatch describes one signature layer that verified.
type signatureMatch struct {
	// KeyPath is the trusted key file that accepted it.
	KeyPath string `json:"keyPath"`
	// Identity is the payload's claimed docker-reference.
	Identity string `json:"identity"`
}

// verifyReport is the JSON `data` payload of llmman_verify.
type verifyReport struct {
	Reference string `json:"reference"`
	Digest    string `json:"digest"`
	Verified  bool   `json:"verified"`
	// SignaturesFound counts signature layers present on the registry,
	// whether or not any of them verified — the difference between
	// "nobody signed this" and "someone signed this, but not with a key
	// you trust", which are very different things to tell a user.
	SignaturesFound int              `json:"signaturesFound"`
	Matches         []signatureMatch `json:"matches"`
	// Reason explains a false Verified. Empty when Verified is true.
	Reason string `json:"reason"`
}

// verifySignatureLayer decides whether one layer is a valid signature,
// by a trusted key, over exactly the manifest asked about, published
// where it claims to be. All four must hold; the error says which
// didn't.
//
// Claim checks run before any key is consulted, but nothing is believed
// until a key accepts the bytes that carried the claims — the ordering
// only buys a cheaper error message. The signature is verified against
// the literal blob bytes, never a re-marshalling of the parsed struct,
// which would normalize away whitespace the signer covered.
func verifySignatureLayer(payload []byte, b64sig, ref string, target digest.Digest, keys []trustedKey) (*signatureMatch, error) {
	if b64sig == "" {
		return nil, fmt.Errorf("layer has no %s annotation", cosignSignatureAnnotation)
	}
	raw, err := base64.StdEncoding.DecodeString(b64sig)
	if err != nil {
		return nil, fmt.Errorf("signature is not valid base64: %w", err)
	}

	var ssp simpleSigningPayload
	if err := json.Unmarshal(payload, &ssp); err != nil {
		return nil, fmt.Errorf("parse payload: %w", err)
	}
	if ssp.Critical.Type != cosignSignatureType {
		return nil, fmt.Errorf("unexpected payload type %q", ssp.Critical.Type)
	}
	if ssp.Critical.Image.DockerManifestDigest != target.String() {
		return nil, fmt.Errorf("payload signs %s, not %s", ssp.Critical.Image.DockerManifestDigest, target)
	}
	if !identityMatchesRepository(ssp.Critical.Identity.DockerReference, ref) {
		return nil, fmt.Errorf("payload claims identity %q, which is not %s",
			ssp.Critical.Identity.DockerReference, repositoryOf(ref))
	}

	for _, key := range keys {
		if err := key.verifier.VerifySignature(bytes.NewReader(raw), bytes.NewReader(payload)); err != nil {
			continue
		}
		return &signatureMatch{
			KeyPath:  key.path,
			Identity: ssp.Critical.Identity.DockerReference,
		}, nil
	}
	return nil, errors.New("no trusted key accepted this signature")
}

// errNoSignature reports that the registry has no signature artifact for
// a digest. Distinguished from a transport failure only for the error
// message's sake: every caller treats both as "not verified", so a
// misclassification here can soften a message but cannot admit an
// unverified model.
var errNoSignature = errors.New("no signature found")

// fetchSignatureManifest returns the signature artifact's manifest plus
// a layout directory holding its blobs, which the caller must remove.
//
// Manifest first, blobs second: it is inspected and bounded
// (checkSignatureManifest) before pullToLayout downloads anything.
// pullToLayout for the blobs keeps this file backend-agnostic.
func fetchSignatureManifest(ctx context.Context, sigRef string) (ocispec.Manifest, string, error) {
	raw, err := fetchManifestRaw(ctx, sigRef)
	if err != nil {
		if isNotFoundError(err) {
			return ocispec.Manifest{}, "", errNoSignature
		}
		return ocispec.Manifest{}, "", fmt.Errorf("fetch signature %s: %w", sigRef, err)
	}
	var m ocispec.Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return ocispec.Manifest{}, "", fmt.Errorf("parse signature manifest: %w", err)
	}
	if err := checkSignatureManifest(&m); err != nil {
		return ocispec.Manifest{}, "", fmt.Errorf("signature %s: %w", sigRef, err)
	}

	// Pinned to the manifest just inspected, not the tag: pullToLayout
	// resolves again, and a tag repointed in between would hand back a
	// different artifact that never passed checkSignatureManifest.
	pinnedRef := repositoryOf(sigRef) + "@" + digest.FromBytes(raw).String()

	tmp, err := os.MkdirTemp("", "llmman-sig-")
	if err != nil {
		return ocispec.Manifest{}, "", fmt.Errorf("create signature staging dir: %w", err)
	}
	// pullToLayout tracks byte progress under the ref it was given (see
	// progress_state.go); nothing polls this one, so drop the entry
	// rather than leaking one per verification in a long-running daemon.
	defer progressDone(pinnedRef)

	if err := pullToLayout(ctx, pinnedRef, tmp); err != nil {
		os.RemoveAll(tmp)
		if isNotFoundError(err) {
			return ocispec.Manifest{}, "", errNoSignature
		}
		return ocispec.Manifest{}, "", fmt.Errorf("fetch signature %s: %w", sigRef, err)
	}
	return m, tmp, nil
}

// checkSignatureManifest rejects an artifact too large or too numerous
// to be a plausible set of signatures, before a byte is fetched.
//
// Every layer, not just cosign-typed ones: pullToLayout fetches all of
// them whatever the media type, so exempting the rest would make the
// budget bypassable by relabelling.
func checkSignatureManifest(m *ocispec.Manifest) error {
	if len(m.Layers) > maxSignatureCount {
		return fmt.Errorf("manifest lists %d layers, over the limit of %d", len(m.Layers), maxSignatureCount)
	}
	total := m.Config.Size
	for _, layer := range m.Layers {
		if layer.Size > maxSignatureBytes {
			return fmt.Errorf("layer %s is %d bytes, over the limit of %d",
				shortDigest(layer.Digest), layer.Size, maxSignatureBytes)
		}
		total += layer.Size
	}
	if total > maxSignatureArtifactBytes {
		return fmt.Errorf("artifact is %d bytes, over the limit of %d", total, maxSignatureArtifactBytes)
	}
	return nil
}

// isNotFoundError reports whether err means "the registry does not have
// this", as opposed to "I could not find out".
//
// Load-bearing on the signing path: signManifest starts from an empty
// layer set when the answer is "absent", so calling a transport failure
// absent would destroy every other signer's signature. Hence the
// strictness — a bare "404" is not enough, since these messages embed
// the digest hex, which contains "404" for one manifest in seventy.
func isNotFoundError(err error) bool {
	if errors.Is(err, errNoSignature) || isBackendNotFound(err) {
		return true
	}
	msg := strings.ToLower(err.Error())
	for _, needle := range []string{
		"manifest unknown",
		"manifest_unknown",
		"name unknown",
		"status 404",
		"status: 404",
		"status code 404",
		"404 not found",
	} {
		if strings.Contains(msg, needle) {
			return true
		}
	}
	return false
}

// verifySignatures checks every signature published for target against
// every trusted key.
//
// A report rather than a bool because the negative outcomes need
// different words: nothing signed, signed by a stranger, or registry
// unreachable. Only the last is an error return — "this model is
// unsigned" is an answer, not a failure to produce one.
func verifySignatures(ctx context.Context, ref string, target digest.Digest, keyPaths []string) (*verifyReport, error) {
	if len(keyPaths) == 0 {
		return nil, errors.New("no trusted public keys configured")
	}
	keys, err := loadTrustedKeys(keyPaths)
	if err != nil {
		return nil, err
	}
	return verifyAgainst(ctx, ref, target, keys)
}

// verifyAgainst is verifySignatures with the keys already loaded, so
// signManifest can ask the same question about its own key.
func verifyAgainst(ctx context.Context, ref string, target digest.Digest, keys []trustedKey) (*verifyReport, error) {
	report := &verifyReport{
		Reference: ref,
		Digest:    target.String(),
		Matches:   []signatureMatch{},
	}

	manifest, dir, err := fetchSignatureManifest(ctx, signatureTagFor(ref, target))
	if errors.Is(err, errNoSignature) {
		report.Reason = describeMissingSignature(ctx, ref, target)
		return report, nil
	}
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)

	// Collected but only surfaced if *nothing* verifies: a manifest
	// carrying signatures from three keys, one of which is ours, is a
	// success, and the other two failing is the expected, uninteresting
	// case — not something to warn about.
	var rejections []string

	for _, layer := range manifest.Layers {
		if layer.MediaType != cosignSignatureMediaType {
			continue
		}
		report.SignaturesFound++

		payload, err := readBlob(dir, layer.Digest)
		if err != nil {
			rejections = append(rejections, fmt.Sprintf("%s: read payload: %v", shortDigest(layer.Digest), err))
			continue
		}
		match, err := verifySignatureLayer(payload, layer.Annotations[cosignSignatureAnnotation], ref, target, keys)
		if err != nil {
			rejections = append(rejections, fmt.Sprintf("%s: %v", shortDigest(layer.Digest), err))
			continue
		}
		report.Matches = append(report.Matches, *match)
	}

	report.Verified = len(report.Matches) > 0
	switch {
	case report.Verified:
	case report.SignaturesFound == 0:
		report.Reason = describeMissingSignature(ctx, ref, target)
	default:
		report.Reason = strings.Join(rejections, "; ")
	}
	return report, nil
}

// describeMissingSignature distinguishes "nothing signed this" from
// "something signed this in a format llmman cannot read".
//
// Current cosign publishes a sigstore *bundle* at a different tag by
// default, so a user who ran `cosign sign` would otherwise be told their
// model is unsigned — false, and misleading. The outcome is unchanged
// either way (still unverified, still refused under enforce); only the
// message differs.
func describeMissingSignature(ctx context.Context, ref string, target digest.Digest) string {
	const absent = "no signature published for this manifest"
	if !hasSigstoreBundle(ctx, ref, target) {
		return absent
	}
	return "this manifest is signed with a sigstore bundle " +
		"(application/vnd.dev.sigstore.bundle), which llmman cannot verify yet — " +
		"it reads cosign's simple-signing format, published at the .sig tag"
}

// hasSigstoreBundle reports whether the suffix-less fallback tag holds a
// sigstore bundle.
//
// Manifest only — never pullToLayout. This runs on an already-failing
// path, purely to sharpen an error message, so it must not become a way
// to make a rejected model cost a second download. Best-effort besides:
// every failure means "no", so a registry hiccup here can only cost a
// less specific message.
func hasSigstoreBundle(ctx context.Context, ref string, target digest.Digest) bool {
	bundleRef := repositoryOf(ref) + ":" + strings.Replace(target.String(), ":", "-", 1)
	raw, err := fetchManifestRaw(ctx, bundleRef)
	if err != nil {
		return false
	}
	// An image index whose entries declare the bundle artifact type —
	// unmarshalled loosely, since all that matters is spotting the type
	// string, not modelling the format llmman doesn't read.
	var index struct {
		Manifests []struct {
			ArtifactType string `json:"artifactType"`
		} `json:"manifests"`
		ArtifactType string `json:"artifactType"`
	}
	if err := json.Unmarshal(raw, &index); err != nil {
		return false
	}
	if strings.HasPrefix(index.ArtifactType, sigstoreBundleArtifactTypePrefix) {
		return true
	}
	for _, m := range index.Manifests {
		if strings.HasPrefix(m.ArtifactType, sigstoreBundleArtifactTypePrefix) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Signing
// ---------------------------------------------------------------------------

// The placeholder config every cosign signature manifest carries: an OCI
// image manifest requires the field, a signature has no configuration to
// describe, and every reader ignores it.
const emptyConfigMediaType = "application/vnd.oci.image.config.v1+json"

var emptyConfig = []byte("{}")

// signManifest signs target with the private key at keyPath and
// publishes a cosign signature artifact alongside ref.
//
// Existing signatures are preserved — a second key appends rather than
// replaces, which is what makes key rotation work. A key that already
// signed this digest does not sign again; detection is by *verifying*
// existing layers against this key's own public half, since ECDSA
// signatures are randomized and the payload blob is identical across
// keys (the signature lives in the annotation), so comparing either
// would be wrong. Mirrors cosign's dupeDetector.
//
// Append is a read-modify-write of a shared tag, and the registry offers
// no compare-and-swap, so this re-reads after pushing and retries if
// what landed is missing something — optimistic concurrency, reporting
// rather than silently losing a signature if it cannot converge.
func signManifest(ctx context.Context, ref string, target digest.Digest, keyPath string, password []byte) error {
	signer, err := signature.LoadSignerFromPEMFile(keyPath, crypto.SHA256, staticPassFunc(password))
	if err != nil {
		return fmt.Errorf("load private key %s: %w", keyPath, err)
	}
	pub, err := signer.PublicKey()
	if err != nil {
		return fmt.Errorf("derive public key from %s: %w", keyPath, err)
	}
	selfVerifier, err := signature.LoadVerifier(pub, crypto.SHA256)
	if err != nil {
		return fmt.Errorf("load verifier for %s: %w", keyPath, err)
	}
	self := []trustedKey{{path: keyPath, verifier: selfVerifier}}

	var payloadDoc simpleSigningPayload
	// The claimed identity is the repository, without a tag: the same
	// digest is reachable under many tags and the signature is over the
	// digest, so naming one tag would be arbitrary and force verifiers
	// to be lenient about it. See identityMatchesRepository.
	payloadDoc.Critical.Identity.DockerReference = repositoryOf(ref)
	payloadDoc.Critical.Image.DockerManifestDigest = target.String()
	payloadDoc.Critical.Type = cosignSignatureType

	payload, err := json.Marshal(payloadDoc)
	if err != nil {
		return fmt.Errorf("marshal signing payload: %w", err)
	}
	raw, err := signer.SignMessage(bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("sign payload: %w", err)
	}
	b64 := base64.StdEncoding.EncodeToString(raw)
	sigRef := signatureTagFor(ref, target)

	const attempts = 3
	for attempt := 0; attempt < attempts; attempt++ {
		pushed, err := appendSignature(ctx, sigRef, ref, target, payload, b64, self)
		if err != nil {
			return err
		}
		if !pushed {
			return nil // this key had already signed it
		}
		// Confirm *this key's* signature is what actually landed, by
		// verifying against it. Counting layers cannot answer that: two
		// signers starting from an empty artifact both push one layer,
		// so the loser sees the winner's count and reads it as its own
		// success, silently losing the signature this loop exists to
		// protect. A false verdict here covers both "overwritten" and
		// "deleted", and both want the same retry.
		landed, err := verifyAgainst(ctx, ref, target, self)
		if err != nil {
			return fmt.Errorf("confirm published signature: %w", err)
		}
		if landed.Verified {
			return nil
		}
	}
	return fmt.Errorf("signature %s kept being overwritten by a concurrent signer after %d attempts", sigRef, attempts)
}

// appendSignature performs one read-modify-write of sigRef's artifact,
// adding a layer carrying payload/b64sig unless one of `self`'s
// signatures is already there. Reports whether it pushed anything.
func appendSignature(
	ctx context.Context,
	sigRef, ref string,
	target digest.Digest,
	payload []byte,
	b64sig string,
	self []trustedKey,
) (pushed bool, err error) {
	// Start from whatever is already published, so this appends. A fetch
	// failure that isn't "absent" is fatal rather than treated as an
	// empty starting point: overwriting an artifact we merely failed to
	// *read* would silently destroy other signers' signatures. See
	// isNotFoundError for how carefully that distinction is drawn.
	var layers []ocispec.Descriptor
	existing, existingDir, err := fetchSignatureManifest(ctx, sigRef)
	switch {
	case err == nil:
		layers = existing.Layers
	case errors.Is(err, errNoSignature):
	default:
		return false, err
	}
	defer func() {
		if existingDir != "" {
			os.RemoveAll(existingDir)
		}
	}()

	staging, err := os.MkdirTemp("", "llmman-sign-")
	if err != nil {
		return false, fmt.Errorf("create signing staging dir: %w", err)
	}
	defer os.RemoveAll(staging)
	if err := ensureLayout(staging); err != nil {
		return false, fmt.Errorf("init signing layout: %w", err)
	}

	// Carry every existing payload blob into the staging layout, or the
	// combined manifest would reference blobs this layout cannot serve.
	// Two keys signing one manifest share a single payload blob and
	// differ only in the annotation, so staging the same content twice
	// is expected; writeBlob is content-addressed.
	for _, layer := range layers {
		blob, err := readBlob(existingDir, layer.Digest)
		if err != nil {
			return false, fmt.Errorf("read existing signature payload %s: %w", layer.Digest, err)
		}
		if _, err := verifySignatureLayer(blob, layer.Annotations[cosignSignatureAnnotation], ref, target, self); err == nil {
			return false, nil // this key already signed this digest
		}
		if _, err := writeBlob(staging, layer.MediaType, blob); err != nil {
			return false, fmt.Errorf("stage existing signature payload %s: %w", layer.Digest, err)
		}
	}
	if len(layers)+1 > maxSignatureCount {
		return false, fmt.Errorf("%s already holds %d signatures, the limit", sigRef, len(layers))
	}

	payloadDesc, err := writeBlob(staging, cosignSignatureMediaType, payload)
	if err != nil {
		return false, fmt.Errorf("stage signature payload: %w", err)
	}
	payloadDesc.Annotations = map[string]string{cosignSignatureAnnotation: b64sig}
	layers = append(layers, payloadDesc)

	configDesc, err := writeBlob(staging, emptyConfigMediaType, emptyConfig)
	if err != nil {
		return false, fmt.Errorf("stage signature config: %w", err)
	}

	manifest := ocispec.Manifest{
		MediaType: ocispec.MediaTypeImageManifest,
		Config:    configDesc,
		Layers:    layers,
	}
	manifest.SchemaVersion = 2
	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return false, fmt.Errorf("marshal signature manifest: %w", err)
	}
	manifestDesc, err := writeBlob(staging, ocispec.MediaTypeImageManifest, manifestJSON)
	if err != nil {
		return false, fmt.Errorf("stage signature manifest: %w", err)
	}
	if err := writeManifestRef(staging, sigRef, manifestDesc); err != nil {
		return false, fmt.Errorf("tag signature manifest: %w", err)
	}
	if _, err := pushToRegistry(ctx, staging, sigRef); err != nil {
		return false, fmt.Errorf("push signature %s: %w", sigRef, err)
	}
	return true, nil
}

// staticPassFunc adapts an already-known passphrase to the callback
// shape sigstore's key loader expects. A nil/empty passphrase is passed
// through as-is, which is correct for an unencrypted PEM (the loader
// never calls back for one).
func staticPassFunc(password []byte) func(bool) ([]byte, error) {
	return func(bool) ([]byte, error) { return password, nil }
}

// ---------------------------------------------------------------------------
// Exported CGO functions
// ---------------------------------------------------------------------------

// llmman_verify checks a manifest's signatures against trusted PEM
// public keys (cKeysJSON, a JSON array of paths).
//
// cDigest may be empty to resolve ref first; passing a known digest
// saves a round trip and removes the window in which a tag could move
// between verification and use.
//
// "Not verified" is success with Verified false; errors mean no answer
// was reached. The caller's policy decides what each means.
//
//export llmman_verify
func llmman_verify(cRef, cDigest, cKeysJSON *C.char) *C.char {
	ref := C.GoString(cRef)
	ctx := withoutMirrors(context.Background())

	var keyPaths []string
	if raw := C.GoString(cKeysJSON); raw != "" {
		if err := json.Unmarshal([]byte(raw), &keyPaths); err != nil {
			return errResp(fmt.Errorf("parse trusted key list: %w", err))
		}
	}

	target, err := targetDigest(ctx, ref, C.GoString(cDigest))
	if err != nil {
		return errResp(err)
	}
	report, err := verifySignatures(ctx, ref, target, keyPaths)
	if err != nil {
		return errResp(err)
	}
	data, err := json.Marshal(report)
	if err != nil {
		return errResp(fmt.Errorf("marshal verification report: %w", err))
	}
	return okResp(string(data))
}

// llmman_sign signs a model manifest and publishes a cosign-format
// signature next to it in the same repository. See signManifest.
//
//export llmman_sign
func llmman_sign(cRef, cDigest, cKeyPath, cPassword *C.char) *C.char {
	ref := C.GoString(cRef)
	ctx := withoutMirrors(context.Background())

	keyPath := C.GoString(cKeyPath)
	if keyPath == "" {
		return errMsg("no signing key given")
	}
	target, err := targetDigest(ctx, ref, C.GoString(cDigest))
	if err != nil {
		return errResp(err)
	}
	if err := signManifest(ctx, ref, target, keyPath, []byte(C.GoString(cPassword))); err != nil {
		return errResp(err)
	}
	return okResp(target.String())
}

// llmman_resolve_digest returns the manifest digest the registry
// currently serves for a reference, without fetching the manifest body
// or any layer. See ffi::resolved_digest_of for why the pull path needs
// this separately from llmman_verify.
//
//export llmman_resolve_digest
func llmman_resolve_digest(cRef *C.char) *C.char {
	ref := C.GoString(cRef)
	if err := notHandledHere(ref); err != nil {
		return errResp(err)
	}
	d, err := resolveManifestDigest(context.Background(), normalizeTag(ref))
	if err != nil {
		return errResp(err)
	}
	return okResp(d.String())
}

// targetDigest returns the manifest digest to sign or verify: the given
// one if the caller already knows it, otherwise whatever ref currently
// resolves to at the registry.
func targetDigest(ctx context.Context, ref, given string) (digest.Digest, error) {
	if given != "" {
		d, err := digest.Parse(given)
		if err != nil {
			return "", fmt.Errorf("parse digest %q: %w", given, err)
		}
		return d, nil
	}
	if err := notHandledHere(ref); err != nil {
		return "", err
	}
	return resolveManifestDigest(ctx, normalizeTag(ref))
}
