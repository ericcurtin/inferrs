//go:build cshared

// C shared library entry point for OCI model operations.
// Built with: go build -buildmode=c-shared -tags cshared
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"fmt"
	"io"
	"os"
	"sync"
	"unsafe"

	"github.com/docker/model-runner/cmd/cli/desktop"
)

// lastError stores the most recent error message.  Protected by a mutex
// for safety, although FFI callers are expected to be single-threaded.
var (
	lastErr   string
	lastErrMu sync.Mutex
)

func setLastError(err error) {
	lastErrMu.Lock()
	defer lastErrMu.Unlock()
	if err != nil {
		lastErr = err.Error()
	} else {
		lastErr = ""
	}
}

// oci_pull pulls an OCI model and returns the bundle path.
// Progress is written to stderr.  Returns NULL on error (retrieve with
// oci_last_error).  Caller must free the returned string with oci_free_string.
//
//export oci_pull
func oci_pull(reference *C.char) *C.char {
	ref := C.GoString(reference)

	client, err := newClient()
	if err != nil {
		setLastError(fmt.Errorf("client: %w", err))
		return nil
	}

	// Create a pipe: PullModel writes JSON progress to pw,
	// DisplayProgress reads from pr and renders Docker-style progress bars.
	pr, pw := io.Pipe()

	type pullResult struct {
		msg      string
		progress bool
		err      error
	}
	done := make(chan pullResult, 1)

	go func() {
		msg, progress, err := desktop.DisplayProgress(pr, &stderrPrinter{})
		done <- pullResult{msg, progress, err}
	}()

	// Pull the model.
	pullErr := client.PullModel(context.Background(), ref, pw)
	pw.Close()

	// Wait for progress rendering to finish.
	result := <-done
	if pullErr != nil {
		setLastError(fmt.Errorf("pull: %w", pullErr))
		return nil
	}
	if result.err != nil {
		setLastError(fmt.Errorf("progress: %w", result.err))
		return nil
	}

	// Print final message from progress stream (e.g. "Model pulled successfully").
	if result.msg != "" {
		fmt.Fprintf(os.Stderr, "%s\n", result.msg)
	}

	// Ensure the bundle is created and return its root directory.
	bundle, err := client.GetBundle(ref)
	if err != nil {
		setLastError(fmt.Errorf("bundle: %w", err))
		return nil
	}

	setLastError(nil)
	return C.CString(bundle.RootDir())
}

// oci_bundle returns the bundle path for an already-pulled model.
// Returns NULL if the model is not in the local store (check oci_last_error
// for details).  Caller must free the returned string with oci_free_string.
//
//export oci_bundle
func oci_bundle(reference *C.char) *C.char {
	ref := C.GoString(reference)

	client, err := newClient()
	if err != nil {
		setLastError(fmt.Errorf("client: %w", err))
		return nil
	}

	bundle, err := client.GetBundle(ref)
	if err != nil {
		setLastError(fmt.Errorf("bundle: %w", err))
		return nil
	}

	setLastError(nil)
	return C.CString(bundle.RootDir())
}

// oci_last_error returns the last error message, or NULL if no error.
// The returned string must be freed by the caller with oci_free_string.
//
//export oci_last_error
func oci_last_error() *C.char {
	lastErrMu.Lock()
	defer lastErrMu.Unlock()
	if lastErr == "" {
		return nil
	}
	return C.CString(lastErr)
}

// oci_free_string frees a string previously returned by oci_pull, oci_bundle,
// or oci_last_error.
//
//export oci_free_string
func oci_free_string(s *C.char) {
	if s != nil {
		C.free(unsafe.Pointer(s))
	}
}

// Empty main required by -buildmode=c-shared.
func main() {}
