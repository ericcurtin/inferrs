//go:build podman

package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/docker/distribution/registry/api/errcode"
	v2 "github.com/docker/distribution/registry/api/v2"

	digest "github.com/opencontainers/go-digest"
	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/vbauerster/mpb/v8"
	"github.com/vbauerster/mpb/v8/decor"
	commonauth "go.podman.io/common/pkg/auth"
	"go.podman.io/image/v5/copy"
	"go.podman.io/image/v5/manifest"
	"go.podman.io/image/v5/signature"
	"go.podman.io/image/v5/transports/alltransports"
	"go.podman.io/image/v5/types"
)

// scratchTag is the tag used inside the throwaway, single-manifest OCI
// layout directories pullToLayout/pushToRegistry hand to go.podman.io/image's
// "oci:" transport (see their own comments) — it never leaves that
// ephemeral directory, so its exact value doesn't matter beyond being a
// valid tag string.
const scratchTag = "llmman-scratch"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func insecurePolicy() (*signature.PolicyContext, error) {
	policy := &signature.Policy{
		Default: signature.PolicyRequirements{
			signature.NewPRInsecureAcceptAnything(),
		},
	}
	return signature.NewPolicyContext(policy)
}

// ---------------------------------------------------------------------------
// Exported CGO functions
// ---------------------------------------------------------------------------

// llmman_login stores credentials for a registry using the go.podman.io/common auth library.
//
//export llmman_login
func llmman_login(cServer, cUsername, cPassword *C.char) *C.char {
	if err := podmanLogin(context.Background(), C.GoString(cServer), C.GoString(cUsername), C.GoString(cPassword)); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// podmanLogin is llmman_login's implementation, factored out so a test
// can reach it — Go forbids cgo in _test.go files.
//
// Stdout must be set: commonauth.Login writes to it unchecked on the
// success path, so leaving it nil panicked *after* the credentials were
// already written. io.Discard, not os.Stdout, because the Rust caller
// (src/cmd/login.rs) prints its own success line.
func podmanLogin(ctx context.Context, server, username, password string) error {
	sys := &types.SystemContext{}
	opts := &commonauth.LoginOptions{
		Username: username,
		Password: password,
		Stdout:   io.Discard,
	}
	if err := commonauth.Login(ctx, sys, opts, []string{server}); err != nil {
		return fmt.Errorf("login: %w", err)
	}
	return nil
}

// llmman_logout removes credentials for a registry.
//
//export llmman_logout
func llmman_logout(cServer *C.char) *C.char {
	if err := podmanLogout(C.GoString(cServer)); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// podmanLogout is llmman_logout's implementation. Factored out, and
// needing Stdout set, for the same reasons as podmanLogin above.
func podmanLogout(server string) error {
	sys := &types.SystemContext{}
	opts := &commonauth.LogoutOptions{
		All:    false,
		Stdout: io.Discard,
	}
	if err := commonauth.Logout(sys, opts, []string{server}); err != nil {
		return fmt.Errorf("logout: %w", err)
	}
	return nil
}

// llmman_push pushes an image from a local OCI layout to a registry.
//
//export llmman_push
func llmman_push(cLayoutDir, cRef *C.char) *C.char {
	ref := C.GoString(cRef)
	progressReset(ref, "retrieving manifest")
	defer progressDone(ref)
	if _, err := pushToRegistry(context.Background(), C.GoString(cLayoutDir), ref); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// pushToRegistry is llmman_push's implementation, factored out from the
// CGO entry point above so it can take and return ordinary Go types.
//
// The push source handed to go.podman.io/image is a throwaway OCI layout
// holding only an index.json naming the one manifest to push — not
// layoutDir itself, whose index.json go.podman.io/image would otherwise
// manage opaquely in exactly the single-shared-file format this whole
// change replaces llmman's long-term store with something else instead
// of. It needs no blob files of its own: sharedBlobDirOpts points the
// source straight at layoutDir/blobs, so every blob (manifest included)
// is read from the real store directly.
func pushToRegistry(ctx context.Context, layoutDir, ref string) (changed bool, err error) {
	manifestDesc, err := findManifestForPush(layoutDir, ref)
	if err != nil {
		return false, err
	}

	scratchDir, cleanup, err := buildScratchOCILayout(layoutDir, manifestDesc)
	if err != nil {
		return false, fmt.Errorf("prepare push source: %w", err)
	}
	defer cleanup()

	srcStr := fmt.Sprintf("oci:%s:%s", scratchDir, scratchTag)
	srcRef, err := alltransports.ParseImageName(srcStr)
	if err != nil {
		return false, fmt.Errorf("parse src ref %q: %w", srcStr, err)
	}

	// Destination: Docker registry. go.podman.io/image's docker transport
	// already defaults a tagless ref to :latest internally, but
	// normalizing here too keeps this consistent with the docker/
	// containerd backend (whose resolver has no such default — see
	// backend_docker.go's pushToRegistry).
	dstStr := "docker://" + normalizeTag(ref)
	dstRef, err := alltransports.ParseImageName(dstStr)
	if err != nil {
		return false, fmt.Errorf("parse dst ref %q: %w", dstStr, err)
	}

	pctx, err := insecurePolicy()
	if err != nil {
		return false, fmt.Errorf("policy context: %w", err)
	}
	defer pctx.Destroy()

	progressSetStatus(ref, "pushing")
	if _, err := copyImageWithProgress(ctx, pctx, dstRef, srcRef, "Pushing", "Pushed", &copy.Options{
		SourceCtx: sharedBlobDirOpts(layoutDir),
	}, &changed, ref); err != nil {
		return false, fmt.Errorf("copy image: %w", err)
	}
	return changed, nil
}

// buildScratchOCILayout assembles a throwaway OCI layout directory whose
// only job is naming manifestDesc under scratchTag for go.podman.io/image's
// "oci:" source transport to resolve — see pushToRegistry's own doc
// comment on why, and on why no blob files need to be staged into it.
// Created inside layoutDir so any same-filesystem operation
// go.podman.io/image performs between it and layoutDir/blobs (see
// sharedBlobDirOpts) can't hit a cross-device error.
func buildScratchOCILayout(layoutDir string, manifestDesc ocispec.Descriptor) (dir string, cleanup func(), err error) {
	dir, err = os.MkdirTemp(layoutDir, ".llmman-push-*")
	if err != nil {
		return "", nil, err
	}
	cleanup = func() { os.RemoveAll(dir) }

	if err := os.WriteFile(filepath.Join(dir, "oci-layout"), []byte(`{"imageLayoutVersion":"1.0.0"}`), 0o644); err != nil {
		cleanup()
		return "", nil, err
	}
	idx := scratchIndex(manifestDesc)
	data, err := json.MarshalIndent(idx, "", "  ")
	if err != nil {
		cleanup()
		return "", nil, err
	}
	if err := os.WriteFile(filepath.Join(dir, "index.json"), data, 0o644); err != nil {
		cleanup()
		return "", nil, err
	}
	return dir, cleanup, nil
}

// sharedBlobDirOpts points go.podman.io/image's "oci:" transport at
// layoutDir/blobs as its shared blob directory: reads/writes of any
// blob (including the manifest itself, which is also content-addressed)
// go straight to/from the real store, skipping any blob already there —
// restoring the pre-scratch-layout "already present"/Cached behavior for
// pulls — without ever staging a copy of it under a scratch directory.
func sharedBlobDirOpts(layoutDir string) *types.SystemContext {
	return &types.SystemContext{OCISharedBlobDirPath: filepath.Join(layoutDir, "blobs")}
}

// llmman_pull pulls an image from a registry into a local OCI layout directory.
//
//export llmman_pull
func llmman_pull(cRef, cLayoutDir *C.char) *C.char {
	ref := C.GoString(cRef)
	progressReset(ref, "pulling manifest")
	defer progressDone(ref)
	if err := pullToLayout(context.Background(), ref, C.GoString(cLayoutDir)); err != nil {
		return errResp(err)
	}
	return okResp("")
}

// pullToLayout is llmman_pull's implementation, factored out so
// llmman_transfer's staging-directory fallback can reuse it.
func pullToLayout(ctx context.Context, ref, layoutDir string) error {
	// progressKey is the exact ref llmman_pull was originally called with
	// (see backend_docker.go's pullToLayout for why this must be
	// captured before classifyPullRef normalizes ref itself).
	progressKey := ref
	ref, err := classifyPullRef(ref)
	if err != nil {
		return err
	}

	if err := ensureLayout(layoutDir); err != nil {
		return fmt.Errorf("init OCI layout: %w", err)
	}

	// Source: Docker registry
	srcStr := "docker://" + ref
	srcRef, err := alltransports.ParseImageName(srcStr)
	if err != nil {
		return fmt.Errorf("parse src ref %q: %w", srcStr, err)
	}

	// Destination: a throwaway scratch OCI layout, not layoutDir itself —
	// see buildScratchOCILayout's doc comment. sharedBlobDirOpts makes
	// every blob (including the manifest) land directly in
	// layoutDir/blobs, reusing anything already cached there instead of
	// re-downloading it, so there's nothing to adopt out of scratchDir
	// afterward and its own index.json is never read.
	scratchDir, err := os.MkdirTemp(layoutDir, ".llmman-pull-*")
	if err != nil {
		return fmt.Errorf("create scratch dir: %w", err)
	}
	defer os.RemoveAll(scratchDir)

	dstStr := fmt.Sprintf("oci:%s:%s", scratchDir, scratchTag)
	dstRef, err := alltransports.ParseImageName(dstStr)
	if err != nil {
		return fmt.Errorf("parse dst ref %q: %w", dstStr, err)
	}

	pctx, err := insecurePolicy()
	if err != nil {
		return fmt.Errorf("policy context: %w", err)
	}
	defer pctx.Destroy()

	progressSetStatus(progressKey, "pulling")
	manifestData, err := copyImageWithProgress(ctx, pctx, dstRef, srcRef, "Pulling", "Pulled", &copy.Options{
		MaxParallelDownloads: 6,
		SourceCtx:            registrySourceCtx(ctx),
		DestinationCtx:       sharedBlobDirOpts(layoutDir),
	}, nil, progressKey)
	if err != nil {
		return fmt.Errorf("copy image: %w", err)
	}

	dgst, err := manifest.Digest(manifestData)
	if err != nil {
		return fmt.Errorf("digest manifest: %w", err)
	}
	manifestDesc := ocispec.Descriptor{
		MediaType: manifest.GuessMIMEType(manifestData),
		Digest:    dgst,
		Size:      int64(len(manifestData)),
	}
	return writeManifestRef(layoutDir, ref, manifestDesc)
}

// scratchIndex builds the minimal, valid OCI image index go.podman.io/image's
// "oci:" transport needs to resolve a single manifest by scratchTag — see
// buildScratchOCILayout.
func scratchIndex(manifestDesc ocispec.Descriptor) ocispec.Index {
	return ocispec.Index{
		Versioned: specs.Versioned{SchemaVersion: 2},
		MediaType: ocispec.MediaTypeImageIndex,
		Manifests: []ocispec.Descriptor{{
			MediaType: manifestDesc.MediaType,
			Digest:    manifestDesc.Digest,
			Size:      manifestDesc.Size,
			Annotations: map[string]string{
				ocispec.AnnotationRefName: scratchTag,
			},
		}},
	}
}

// copyImageMu serializes the actual copy.Image call across every
// concurrent pull/push in this process (podman build only).
//
// Unlike the docker/containerd backend (backend_docker.go), which fetches
// and writes each blob itself and can therefore deduplicate concurrent
// fetches of the very same digest via blobFetchGroup (see its own doc
// comment), copy.Image is a single opaque call into go.podman.io/image:
// there's no hook to intercept its internal per-blob writes into the OCI
// layout's blobs/ directory. Now that pulls/pushes of *different* models
// run concurrently (see the Rust daemon's per-model lock registry), two
// such calls could race to write the exact same shared blob at once with
// no way for this package to arbitrate between them. Rather than risk
// that corruption, the podman build keeps this one step — actual data
// transfer — fully serialized, while still letting everything else about
// two concurrent pulls (manifest resolution, HTTP auth, local store
// checks) proceed in parallel. This is more conservative than strictly
// necessary (it also serializes two pulls that share no blobs at all),
// but correctness first: see the docker backend for the finer-grained
// alternative used where it's actually achievable.
var copyImageMu sync.Mutex

// copyImageWithProgress runs copyImageAttempt with retry and stall
// detection, via the same retryStream helper (shared_oci.go) that already
// backs transfer_docker.go's streaming pushes — for the one path that had
// none of it: a real registry pull that simply
// stalled mid-blob-download (zero bytes, indefinitely, no error) is
// exactly what a plain context.Context with no deadline of its own can't
// recover from, and it's what first surfaced this gap — the new
// podman-backend e2e CI coverage's very first run hit a genuine 600s
// test-harness timeout here with zero visible progress. isHTTP4xx stops
// retrying immediately on a permanent error (bad ref, auth failure, ...)
// rather than wasting up to dlMaxAttempts backoff cycles on one.
//
// A retry after a stall (or any other transient error) simply calls
// copy.Image again from scratch: neither of its two destinations (a local
// OCI layout directory for pulls, a registry for pushes) loses
// already-completed blobs between attempts, so a retry only re-fetches
// whatever didn't finish the first time — the same "retry, don't resume"
// trade-off shared_oci.go's own doc comment already accepts for
// transfer_docker.go's non-resumable registry-push path, for the same
// underlying reason: copy.Image (like that path) has no protocol-level
// way to resume a partial blob.
// The returned []byte is copy.Image's own copiedManifest — the exact
// bytes it wrote to dst — which pullToLayout uses directly instead of
// ever reading dst's own index.json back out again (see
// adoptScratchPull). Callers that don't need it (pushToRegistry,
// podmanTransferOCI) simply discard it.
func copyImageWithProgress(ctx context.Context, pctx *signature.PolicyContext, dst, src types.ImageReference, present, pastTense string, opts *copy.Options, changed *bool, progressKey string) ([]byte, error) {
	// Held for the whole retry sequence, not just one attempt — see this
	// mutex's own doc comment on why the actual data transfer is kept
	// fully serialized across every concurrent pull/push in this process.
	copyImageMu.Lock()
	defer copyImageMu.Unlock()

	// copy.Image fetches the manifest and every blob inside this one
	// retryStream, so the manifest phase is not separately bounded to a
	// single attempt the way the Rust HF metadata GET is (hf::client::once). A
	// typo'd but syntactically valid OCI registry host can still run the
	// full retry budget on its manifest lookup here: a known residual, left
	// as a follow-up. Malformed refs never reach this path: validate_reference
	// (Rust, Layer B) rejects them up front.
	var manifestData []byte
	err := retryStream(ctx, progressKey, isHTTP4xx, func() error {
		data, err := copyImageAttempt(ctx, pctx, dst, src, present, pastTense, opts, changed, progressKey)
		if err == nil {
			manifestData = data
		}
		return err
	})
	return manifestData, err
}

// copyImageAttempt runs a single copy.Image call with an mpb bar per
// artifact (for direct/foreground FFI callers, e.g. `llmman transfer`'s
// podman backend — though transfer_podman.go's own copy.Image calls
// don't currently go through this, only pull/push do) and folds the
// same byte counts into progressKey's entry in the shared progressState
// snapshot (see progress_state.go) that lets cmd::serve poll them out of
// the daemon process — two consumers of the same underlying
// go.podman.io/image progress channel. present/pastTense label each
// artifact's bar (e.g. "Pulling"/"Pulled", "Pushing"/"Pushed").
//
// If changed is non-nil, it's set to true whenever at least one artifact
// actually completes a copy (types.ProgressEventDone) rather than turning
// out to already exist at the destination (types.ProgressEventSkipped,
// which never leads to a Done for that same artifact) — letting a caller
// like pushToRegistry/podmanTransferOCI tell whether anything was really
// pushed, e.g. to report "already up to date" for a no-op re-transfer.
//
// Stall detection: copy.Image is a single opaque call into
// go.podman.io/image with no per-blob hooks of its own — unlike a plain
// HTTP download (which can wrap its own response body in a stallReader),
// the only signal available here is copy.Image's own
// Progress channel, so a ProgressEventRead/NewArtifact/Done/Skipped
// callback IS this backend's only "still alive" signal. The watchdog
// goroutine below cancels a context derived from ctx (independent of
// whatever cancellation the caller's own ctx already provides) if
// dlStallTimeout passes with no such callback at all — which, since it
// covers the time before the first callback too, also catches a stall
// during manifest/credential resolution, not just mid-blob-download.
//
// Tracked *per artifact*, not as one process-wide "any event at all"
// clock: MaxParallelDownloads (see pullToLayout) means several artifacts
// can be in flight at once, and an image with more artifacts than that
// cap queues the rest behind whichever ones are currently running. A
// single shared clock reset by *any* artifact's event can never notice
// one specific artifact wedged indefinitely (dead connection, stuck
// mid-read) as long as its concurrent siblings keep producing events of
// their own — exactly the failure mode a real podman-backend e2e run hit
// after the original single-clock version of this watchdog shipped: the
// job's own diagnostic step showed every blob already sitting at its
// full expected size on disk, yet the pull never returned, which a
// per-artifact check (rather than the aggregate one) is what's actually
// needed to catch.
func copyImageAttempt(ctx context.Context, pctx *signature.PolicyContext, dst, src types.ImageReference, present, pastTense string, opts *copy.Options, changed *bool, progressKey string) ([]byte, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	prog := mpb.New(mpb.WithOutput(os.Stderr))
	ch := make(chan types.ProgressProperties)
	bars := make(map[string]*mpb.Bar)
	progDone := make(chan struct{})

	var mu sync.Mutex
	callStart := time.Now()
	// Last-event time for each artifact copy.Image has told us about via
	// ProgressEventNewArtifact but not yet finished (Done/Skipped) —
	// entries are removed on completion so a finished artifact can never
	// be (mis)reported as the stalled one.
	openArtifacts := make(map[string]time.Time)
	var stalledArtifact string // set once, only when the watchdog actually fires

	go func() {
		defer close(progDone)
		for p := range ch {
			key := p.Artifact.Digest.String()
			now := time.Now()

			switch p.Event {
			case types.ProgressEventNewArtifact:
				mu.Lock()
				openArtifacts[key] = now
				mu.Unlock()

				total := p.Artifact.Size
				if total < 0 {
					total = 0
				}
				progressAddTotal(progressKey, total)
				short := p.Artifact.Digest.Hex()
				if len(short) > 12 {
					short = short[:12]
				}
				bar := prog.AddBar(total,
					mpb.BarFillerClearOnComplete(),
					mpb.PrependDecorators(
						decor.OnComplete(decor.Name(present+"  "+short), pastTense+"   "+short),
					),
					mpb.AppendDecorators(
						decor.OnComplete(decor.CountersKibiByte("% .1f / % .1f"), ""),
						decor.OnComplete(decor.Name("  "), ""),
						decor.OnComplete(decor.AverageSpeed(decor.SizeB1024(0), "% .1f"), ""),
					),
				)
				if total == 0 {
					bar.SetTotal(0, true)
				}
				bars[key] = bar
			case types.ProgressEventRead:
				mu.Lock()
				if _, ok := openArtifacts[key]; ok {
					openArtifacts[key] = now
				}
				mu.Unlock()

				progressAddCompleted(progressKey, int64(p.OffsetUpdate))
				if bar, ok := bars[key]; ok {
					bar.IncrInt64(int64(p.OffsetUpdate))
				}
			case types.ProgressEventDone:
				mu.Lock()
				delete(openArtifacts, key)
				mu.Unlock()

				// p.OffsetUpdate here is whatever portion of this
				// artifact's bytes hadn't yet been reported by a Read
				// event above (see progressReader.reportDone in
				// go.podman.io/image's copy/progress_channel.go: Offset
				// is the cumulative total, OffsetUpdate is the remainder
				// since the last update) — the same field, same meaning,
				// as ProgressEventRead's own OffsetUpdate just above.
				// Without this, any transfer that finishes inside a
				// single 200ms ProgressInterval tick (small blobs
				// routinely do) never fires a single Read event at all —
				// only NewArtifact (sets total) then straight to Done —
				// so completed silently stayed at 0 for that artifact
				// forever, however long the rest of the pull took: a
				// correctly-sized, permanently-empty-looking bar, not
				// merely a delayed one.
				progressAddCompleted(progressKey, int64(p.OffsetUpdate))

				if changed != nil {
					*changed = true
				}
				if bar, ok := bars[key]; ok {
					// SetTotal with triggerComplete forces current=total regardless of
					// timing, then fires done() — the OnComplete decorators take over.
					bar.SetTotal(int64(p.Offset), true)
					delete(bars, key)
				}
			case types.ProgressEventSkipped:
				mu.Lock()
				delete(openArtifacts, key)
				mu.Unlock()

				// This artifact turned out to already exist at the
				// destination, so go.podman.io/image never routed it
				// through copyBlobFromStream/newProgressReader at all —
				// see copy/single.go: the "reused" short-circuit fires
				// ProgressEventSkipped directly and returns *before* ever
				// reaching the code that fires ProgressEventNewArtifact.
				// So, unlike every other case in this switch, total was
				// *never* incremented for this artifact — there is no
				// earlier addition to undo (an older version of this got
				// that backwards and subtracted the size back out here,
				// which could only ever under- or, worse, over-correct
				// total for whatever this pull's other artifacts had
				// already added it to) — both total and completed need
				// to be credited with the full size right here, together,
				// as the only place this artifact's size is ever
				// accounted for at all.
				//
				// Why this matters in practice: llmman's local store is
				// content-addressed, so the very same model blob is
				// frequently already present under a *different*
				// reference (e.g. the same GGUF re-pulled as
				// `docker.io/ai/qwen3.5:0.8b` after already having it as
				// `hf.co/unsloth/Qwen3.5-0.8B-GGUF`) — a routine case for
				// llmman, not a rare edge case, and exactly what a fresh
				// `llmman pull qwen3.5:0.8b` hits every single blob of.
				// Without crediting total here, a pull whose only
				// non-skipped artifact was the tiny manifest/config blob
				// (a few hundred bytes) left total pinned at that few
				// hundred bytes for the *entire* operation while gigabytes
				// of skipped weights were credited only to completed —
				// which cmd::serve's relay (stream_ffi_progress) then
				// clamps back down to that same tiny total before it ever
				// reaches the Rust CLI, so `llmman pull`/`run`/`launch`
				// showed a bar frozen at "327 B / 327 B" for however long
				// the skip/verification took, with no sign the real,
				// much larger transfer (or non-transfer) was happening at
				// all. Crediting both here keeps total an accurate
				// picture of the whole pull's bytes while marking this
				// artifact's share instantly done, exactly like a real
				// transfer that finished the moment it started.
				progressAddTotal(progressKey, p.Artifact.Size)
				progressAddCompleted(progressKey, p.Artifact.Size)
				if bar, ok := bars[key]; ok {
					bar.Abort(true)
					delete(bars, key)
					fmt.Fprintf(prog, "Cached   %s\n", p.Artifact.Digest.Hex()[:12])
				}
			}
		}
	}()

	watchdogDone := make(chan struct{})
	go func() {
		defer close(watchdogDone)
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				mu.Lock()
				stale := ""
				if len(openArtifacts) == 0 {
					// Nothing opened yet at all (manifest/credential
					// resolution, or an image with zero artifacts) — fall
					// back to the call's own start time as the baseline.
					if time.Since(callStart) > dlStallTimeout {
						stale = "<manifest/credential resolution>"
					}
				} else {
					for key, last := range openArtifacts {
						if time.Since(last) > dlStallTimeout {
							stale = key
							break
						}
					}
				}
				if stale != "" {
					stalledArtifact = stale
				}
				mu.Unlock()
				if stale != "" {
					cancel()
					return
				}
			}
		}
	}()

	opts.Progress = ch
	opts.ProgressInterval = 200 * time.Millisecond

	manifestData, err := copy.Image(ctx, pctx, dst, src, opts)
	close(ch)
	<-progDone

	// Stop the watchdog goroutine now that the real work is done, instead
	// of leaving it running: its own select loop only ever exits via
	// <-ctx.Done() or by detecting genuine staleness itself (which takes
	// the *full* dlStallTimeout, however fast copy.Image above actually
	// returned) — and until this explicit cancel, nothing ever fired
	// ctx.Done() this early: the `defer cancel()` at the top of this
	// function only runs once copyImageAttempt itself returns, which is
	// blocked right here on <-watchdogDone. Without this line, *every*
	// pull — even one that copy.Image finishes in well under a second,
	// e.g. because every blob was already on disk under a different
	// reference (llmman's local store is content-addressed, so this is
	// routine, not rare) — unconditionally blocked for a further
	// dlStallTimeout (60s) after the real work was already done, waiting
	// for the watchdog's own periodic staleness check to eventually
	// notice "no artifacts open, unclear why" and cancel itself. That
	// full 60s (near enough to the ~60-65s delays seen end-to-end,
	// however the pull's actual bytes played out) was pure dead time:
	// `llmman pull qwen3.5:0.8b` looked "stuck" at a 100%-complete
	// progress bar for about a minute before ever printing "success".
	cancel()
	<-watchdogDone

	// The real root cause of a hang that was chased through several
	// earlier attempts here (single shared stall clock, then a per-
	// artifact one): none of that was ever the problem. A goroutine dump
	// of a real hung local repro showed copy.Image above had *already
	// returned successfully* — the pull had fully completed, every blob
	// correctly written, index.json correctly committed — and the caller
	// was blocked forever in prog.Wait() below instead, because 3 of the
	// 4 bars in `bars` were still alive, waiting in mpb's own Bar.serve
	// select loop: go.podman.io/image does not reliably send a
	// ProgressEventDone/Skipped for every ProgressEventNewArtifact it
	// sends (confirmed empirically, not documented).
	//
	// bar.Abort() does not reliably fix this either (also confirmed with
	// a second repro, after adding it): Abort's own doc comment says it
	// "interrupts bar's running goroutine", but internally (bar.go's
	// done()) that only calls bar.cancel() when the bar isn't
	// auto-refreshing — when it is (the default here, since none of
	// AddBar's options below disable it), it instead just schedules an
	// async "early refresh" that isn't guaranteed to actually observe
	// the abort and unblock serve()'s select loop in every case, e.g.
	// once nothing is left driving further renders. mpb's own
	// Progress.Wait() has no timeout of its own — it simply blocks on a
	// WaitGroup until every bar it was ever given reports done, with no
	// way to force that from outside short of cancelling each bar's
	// context, which prog := mpb.New(...) above never gave us a handle
	// to (mpb.New, not mpb.NewWithContext).
	//
	// Given none of that is fixable from here without depending on mpb
	// internals more deeply than its own public API supports, treat
	// Wait() itself as untrustworthy and never let it block this
	// function: prog was created with mpb.WithOutput(os.Stderr), which
	// for `llmman serve`'s daemon is a redirected log file, not a live
	// terminal (see daemon.rs), so these bars' own rendering has no
	// real audience to begin with. Giving up on waiting for their
	// cleanup after a short grace period and leaking their now-idle
	// goroutines is strictly safer than blocking the entire pull/push on
	// a cosmetic detail that copy.Image's own success or failure above
	// never depended on.
	progWaitDone := make(chan struct{})
	go func() {
		defer close(progWaitDone)
		prog.Wait()
	}()
	select {
	case <-progWaitDone:
	case <-time.After(5 * time.Second):
		fmt.Fprintf(os.Stderr, "[llmman] warning: giving up waiting on progress-bar cleanup after 5s (cosmetic only, not a real stall)\n")
	}
	mu.Lock()
	stalled := stalledArtifact
	mu.Unlock()
	if err != nil && stalled != "" {
		return nil, fmt.Errorf("stalled: artifact %s made no progress for over %v: %w", stalled, dlStallTimeout, err)
	}
	return manifestData, err
}

// resolveManifestDigest returns the manifest digest the registry
// currently serves for ref. See backend_docker.go's counterpart for why
// this is the one part of signature handling each backend has to
// implement itself.
//
// go.podman.io/image has no digest-only lookup, so unlike containerd's
// HEAD-and-read-the-header this actually reads the manifest body — a few
// hundred bytes, and no layer is touched either way.
func resolveManifestDigest(ctx context.Context, ref string) (digest.Digest, error) {
	raw, err := fetchManifestRaw(ctx, ref)
	if err != nil {
		return "", err
	}
	return manifest.Digest(raw)
}

// fetchManifestRaw returns ref's manifest bytes, without fetching any
// layer. See backend_docker.go's counterpart.
func fetchManifestRaw(ctx context.Context, ref string) ([]byte, error) {
	srcStr := "docker://" + ref
	srcRef, err := alltransports.ParseImageName(srcStr)
	if err != nil {
		return nil, fmt.Errorf("parse ref %q: %w", srcStr, err)
	}
	src, err := srcRef.NewImageSource(ctx, registrySourceCtx(ctx))
	if err != nil {
		return nil, fmt.Errorf("open image source: %w", err)
	}
	defer src.Close()
	// nil instanceDigest: the top-level manifest.
	data, _, err := src.GetManifest(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("fetch manifest: %w", err)
	}
	if int64(len(data)) > maxManifestBytes {
		return nil, fmt.Errorf("manifest is %d bytes, over the %d-byte limit", len(data), maxManifestBytes)
	}
	return data, nil
}

// llmman_inspect fetches and returns the raw manifest JSON for a remote reference.
//
//export llmman_inspect
func llmman_inspect(cRef *C.char) *C.char {
	manifestData, err := fetchManifestRaw(context.Background(), C.GoString(cRef))
	if err != nil {
		return errResp(err)
	}

	var buf bytes.Buffer
	if err := json.Indent(&buf, manifestData, "", "  "); err != nil {
		return okResp(string(manifestData))
	}
	return okResp(buf.String())
}

// llmman_transfer transfers an image directly from source to destination,
// without ever writing it to the persistent local store. See
// transfer_podman.go for what this picks between (a direct
// docker://→docker:// copy.Image, which streams every blob straight
// through since go.podman.io/image already knows each one's digest from
// the source manifest; or a staging-directory fallback for HuggingFace
// and other non-OCI sources, which go.podman.io/image has no source
// transport for).
//
//export llmman_transfer
func llmman_transfer(cSource, cDestination *C.char) *C.char {
	changed, pushed, err := podmanTransfer(context.Background(), C.GoString(cSource), C.GoString(cDestination))
	if err != nil {
		return errResp(err)
	}
	// See transferOutcome (transfer_common.go) for what data carries.
	return okResp(transferResultJSON(changed, pushed))
}

// isBackendNotFound reports go.podman.io/image's typed "manifest
// unknown", which docker_client.go recognizes the same way. See
// sigstore.go's isNotFoundError for why a typed answer matters.
func isBackendNotFound(err error) bool {
	var ec errcode.ErrorCoder
	return errors.As(err, &ec) &&
		(ec.ErrorCode() == v2.ErrorCodeManifestUnknown || ec.ErrorCode() == v2.ErrorCodeNameUnknown)
}
