/*
 * stub.c — fallback implementation of the libocipull FFI surface.
 *
 * Used on platforms where the Go CGO toolchain cannot produce a C shared
 * library (e.g. Windows ARM64).  Every function returns NULL / an error so
 * OCI operations fail gracefully at runtime with a clear message instead of
 * a link-time or load-time crash.
 *
 * Exports the same four symbols as the real Go library (lib.go):
 *   oci_pull, oci_bundle, oci_last_error, oci_free_string
 */

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static const char *STUB_ERROR =
    "OCI model operations are not available on this platform "
    "(libocipull was built as a stub)";

/* Pull an OCI model — always fails on stub builds. */
EXPORT char *oci_pull(const char *reference) {
    (void)reference;
    return NULL;
}

/* Look up a cached bundle — always fails on stub builds. */
EXPORT char *oci_bundle(const char *reference) {
    (void)reference;
    return NULL;
}

/* Return the last error message. */
EXPORT char *oci_last_error(void) {
    char *msg = malloc(strlen(STUB_ERROR) + 1);
    if (msg) {
        strcpy(msg, STUB_ERROR);
    }
    return msg;
}

/* Free a string returned by the above functions. */
EXPORT void oci_free_string(char *s) {
    free(s);
}
