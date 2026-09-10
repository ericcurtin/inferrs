//go:build !podman

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/containerd/containerd/v2/core/remotes/docker"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

// requestLog records what a registry was asked, so a test can tell
// whether a pull went to the mirror or to the registry itself.
type requestLog struct {
	mu   sync.Mutex
	reqs []string // "METHOD /path"
}

func (l *requestLog) record(r *http.Request) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.reqs = append(l.reqs, r.Method+" "+r.URL.Path)
}

func (l *requestLog) count(method, pathPrefix string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := 0
	for _, req := range l.reqs {
		m, p, _ := strings.Cut(req, " ")
		if (method == "" || m == method) && strings.HasPrefix(p, pathPrefix) {
			n++
		}
	}
	return n
}

// startLoggedRegistry is startRegistryWithFault that also records every
// request. fail, when set, makes the registry answer 503 to everything
// (reachable but broken) — how a mirror that is down looks to a pull.
func startLoggedRegistry(t *testing.T, log *requestLog, fail func() bool) string {
	t.Helper()
	return startRegistryWithFault(t, func(w http.ResponseWriter, r *http.Request) bool {
		log.record(r)
		if fail != nil && fail() {
			http.Error(w, "mirror is down", http.StatusServiceUnavailable)
			return true
		}
		return false
	})
}

// withMirror configures mirror as the one mirror of registry for the
// rest of the test.
func withMirror(t *testing.T, registry, mirror string) {
	t.Helper()
	cfg, _ := json.Marshal(map[string][]string{registry: {"http://" + mirror}})
	if err := setRegistryMirrors(string(cfg)); err != nil {
		t.Fatalf("configure mirror: %v", err)
	}
	t.Cleanup(func() { _ = setRegistryMirrors("") })
}

func TestMirroredHostsPutsMirrorsFirstWithPullOnlyCapabilities(t *testing.T) {
	t.Cleanup(func() { _ = setRegistryMirrors("") })
	if err := setRegistryMirrors(`{"docker.io": ["https://mirror.gcr.io", "http://cache.corp:5000/registry"]}`); err != nil {
		t.Fatalf("set: %v", err)
	}
	defaults := func(host string) ([]docker.RegistryHost, error) {
		return []docker.RegistryHost{{
			Host:         "registry-1.docker.io",
			Scheme:       "https",
			Path:         "/v2",
			Capabilities: docker.HostCapabilityPull | docker.HostCapabilityResolve | docker.HostCapabilityPush,
		}}, nil
	}
	hosts, err := mirroredHosts(defaults)("docker.io")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"https://mirror.gcr.io/v2", "http://cache.corp:5000/registry/v2", "https://registry-1.docker.io/v2"}
	if len(hosts) != len(want) {
		t.Fatalf("got %d hosts, want %d: %+v", len(hosts), len(want), hosts)
	}
	for i, h := range hosts {
		if got := fmt.Sprintf("%s://%s%s", h.Scheme, h.Host, h.Path); got != want[i] {
			t.Errorf("host %d = %s, want %s", i, got, want[i])
		}
		isMirror := i < len(want)-1
		if isMirror && h.Capabilities.Has(docker.HostCapabilityPush) {
			t.Errorf("mirror %s can be pushed to", h.Host)
		}
		if isMirror && !h.Capabilities.Has(docker.HostCapabilityPull|docker.HostCapabilityResolve) {
			t.Errorf("mirror %s cannot resolve and pull", h.Host)
		}
		if !isMirror && !h.Capabilities.Has(docker.HostCapabilityPush) {
			t.Errorf("the registry itself lost its push capability")
		}
	}

	// An unmirrored registry is passed through untouched.
	hosts, err = mirroredHosts(defaults)("ghcr.io")
	if err != nil || len(hosts) != 1 || hosts[0].Host != "registry-1.docker.io" {
		t.Errorf("unmirrored host list changed: %+v, %v", hosts, err)
	}
}

func TestPullPrefersTheMirrorAndFallsBackToTheRegistry(t *testing.T) {
	ctx := context.Background()
	var upstreamLog, mirrorLog requestLog
	upstream := startLoggedRegistry(t, &upstreamLog, nil)
	mirror := startLoggedRegistry(t, &mirrorLog, nil)
	withMirror(t, upstream, mirror)

	// Only the registry has the model: the mirror must be asked first,
	// answer 404, and the pull must then succeed from the registry.
	ref := upstream + "/org/model:v1"
	want := publishModel(t, ref)
	upstreamLog.reqs, mirrorLog.reqs = nil, nil

	dst := t.TempDir()
	if err := pullToLayout(ctx, ref, dst); err != nil {
		t.Fatalf("pull with an empty mirror: %v", err)
	}
	if got := mirrorLog.count(http.MethodHead, "/v2/org/model/manifests/"); got == 0 {
		t.Errorf("the mirror was never asked to resolve the tag: %v", mirrorLog.reqs)
	}
	if got := upstreamLog.count("", "/v2/org/model/"); got == 0 {
		t.Errorf("the registry was never fallen back to: %v", upstreamLog.reqs)
	}
	assertLayoutHas(t, dst, ref, want)

	// Now the mirror has the model too (published under the same
	// repository, as a pull-through cache would hold it) and the
	// registry is broken: the pull must complete from the mirror alone.
	publishModel(t, mirror+"/org/model:v1")
	broken := startLoggedRegistry(t, &upstreamLog, func() bool { return true })
	// Re-point the mirror at the broken registry's name: the reference
	// host is what the mirror is looked up by.
	withMirror(t, broken, mirror)
	publishBrokenRef := broken + "/org/model:v1"
	upstreamLog.reqs, mirrorLog.reqs = nil, nil

	dst = t.TempDir()
	if err := pullToLayout(ctx, publishBrokenRef, dst); err != nil {
		t.Fatalf("pull from the mirror with the registry down: %v", err)
	}
	if got := mirrorLog.count(http.MethodGet, "/v2/org/model/blobs/"); got == 0 {
		t.Errorf("no blob came from the mirror: %v", mirrorLog.reqs)
	}
	assertLayoutHas(t, dst, publishBrokenRef, want)
}

func TestPullSurvivesAMirrorThatIsDown(t *testing.T) {
	ctx := context.Background()
	var upstreamLog, mirrorLog requestLog
	upstream := startLoggedRegistry(t, &upstreamLog, nil)
	mirror := startLoggedRegistry(t, &mirrorLog, func() bool { return true })
	withMirror(t, upstream, mirror)

	ref := upstream + "/org/model:v1"
	want := publishModel(t, ref)
	dst := t.TempDir()
	if err := pullToLayout(ctx, ref, dst); err != nil {
		t.Fatalf("pull with a 503ing mirror: %v", err)
	}
	if mirrorLog.count("", "/v2/") == 0 {
		t.Error("the down mirror was never even tried")
	}
	assertLayoutHas(t, dst, ref, want)

	// And one nothing is listening on at all.
	withMirror(t, upstream, "127.0.0.1:1")
	dst = t.TempDir()
	if err := pullToLayout(ctx, ref, dst); err != nil {
		t.Fatalf("pull with an unreachable mirror: %v", err)
	}
	assertLayoutHas(t, dst, ref, want)
}

func TestPushIgnoresMirrors(t *testing.T) {
	var upstreamLog, mirrorLog requestLog
	upstream := startLoggedRegistry(t, &upstreamLog, nil)
	mirror := startLoggedRegistry(t, &mirrorLog, nil)
	withMirror(t, upstream, mirror)

	ref := upstream + "/org/model:v1"
	publishModel(t, ref) // pushes through pushToRegistry
	if got := mirrorLog.count("", "/v2/org/"); got != 0 {
		t.Errorf("the mirror saw %d push-side requests: %v", got, mirrorLog.reqs)
	}
	if got := upstreamLog.count(http.MethodPut, "/v2/org/model/manifests/"); got == 0 {
		t.Errorf("the manifest was not pushed to the registry: %v", upstreamLog.reqs)
	}
}

func TestResolveDigestAndInspectUseTheMirror(t *testing.T) {
	ctx := context.Background()
	var upstreamLog, mirrorLog requestLog
	broken := startLoggedRegistry(t, &upstreamLog, func() bool { return true })
	mirror := startLoggedRegistry(t, &mirrorLog, nil)
	withMirror(t, broken, mirror)

	want := publishModel(t, mirror+"/org/model:v1")
	ref := broken + "/org/model:v1"
	got, err := resolveManifestDigest(ctx, ref)
	if err != nil {
		t.Fatalf("resolve through the mirror: %v", err)
	}
	if got != want {
		t.Errorf("resolved %s, want %s", got, want)
	}
	raw, err := fetchManifestRaw(ctx, ref)
	if err != nil {
		t.Fatalf("inspect through the mirror: %v", err)
	}
	var m ocispec.Manifest
	if err := json.Unmarshal(raw, &m); err != nil || len(m.Layers) != 1 {
		t.Errorf("inspect returned %q (%v)", raw, err)
	}
}

// Signatures are read and written at the registry, never a mirror: a
// stale mirror would otherwise have appendSignature rebuild the
// signature manifest without other signers' newer signatures.
func TestSignaturesBypassMirrors(t *testing.T) {
	ctx := context.Background()
	var upstreamLog, mirrorLog requestLog
	upstream := startLoggedRegistry(t, &upstreamLog, nil)
	mirror := startLoggedRegistry(t, &mirrorLog, nil)
	withMirror(t, upstream, mirror)

	ref := upstream + "/org/model:v1"
	target := publishModel(t, ref)
	privPath, pubPath := writeKeyPair(t, t.TempDir(), "signer")
	mirrorLog.reqs = nil

	signCtx := withoutMirrors(ctx)
	if err := signManifest(signCtx, ref, target, privPath, nil); err != nil {
		t.Fatalf("sign: %v", err)
	}
	report, err := verifySignatures(signCtx, ref, target, []string{pubPath})
	if err != nil || !report.Verified {
		t.Fatalf("verify: %v, %+v", err, report)
	}
	if got := mirrorLog.count("", "/v2/"); got != 0 {
		t.Errorf("signing and verifying sent %d requests to the mirror: %v", got, mirrorLog.reqs)
	}
	// And the plain context does use it, so the test is telling them apart.
	if _, err := resolveManifestDigest(ctx, ref); err != nil {
		t.Fatal(err)
	}
	if mirrorLog.count(http.MethodHead, "/v2/org/model/manifests/") == 0 {
		t.Error("a mirrored resolve never reached the mirror")
	}
}

// assertLayoutHas checks that ref in layoutDir names the manifest
// digest want and that its blobs are all present.
func assertLayoutHas(t *testing.T, layoutDir, ref string, want interface{ String() string }) {
	t.Helper()
	desc, err := readManifestRef(layoutDir, ref)
	if err != nil {
		t.Fatalf("pulled layout has no %s: %v", ref, err)
	}
	if desc.Digest.String() != want.String() {
		t.Fatalf("pulled %s, want %s", desc.Digest, want)
	}
	raw, err := readBlob(layoutDir, desc.Digest)
	if err != nil {
		t.Fatalf("manifest blob missing: %v", err)
	}
	var m ocispec.Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("parse pulled manifest: %v", err)
	}
	for _, d := range append([]ocispec.Descriptor{m.Config}, m.Layers...) {
		if _, err := readBlob(layoutDir, d.Digest); err != nil {
			t.Errorf("blob %s missing after pull: %v", d.Digest, err)
		}
	}
}
