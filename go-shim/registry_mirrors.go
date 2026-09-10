package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strings"
	"sync"
)

// Registry mirrors: endpoints a pull tries, in order, before the registry
// itself, falling through on 404 or a connection failure. Pushes and
// signatures always go to the registry. The Rust side owns the
// configuration (src/config.rs registry_mirrors) and hands it over
// through llmman_set_registry_mirrors before its first registry call;
// each backend wires it in — mirroredHosts (backend_docker.go),
// registrySourceCtx (registries_podman.go).

// mirror is one parsed endpoint: scheme, host[:port], and an optional
// path prefix without trailing slash ("" or "/registry").
type mirror struct {
	scheme, host, path string
}

func (m mirror) String() string { return m.scheme + "://" + m.host + m.path }

// mirrorSet maps a canonical registry host to its mirrors, in order.
type mirrorSet map[string][]mirror

var registryMirrors struct {
	mu     sync.RWMutex
	byHost mirrorSet
}

// canonicalRegistryHost folds Docker Hub's names onto "docker.io", which
// is what a reference names and what containerd's RegistryHosts callback
// is asked for.
func canonicalRegistryHost(host string) string {
	host = strings.ToLower(strings.TrimSpace(host))
	switch host {
	case "index.docker.io", "registry-1.docker.io":
		return "docker.io"
	}
	return host
}

// parseMirror accepts `[scheme://]host[:port][/path]`, https by default.
// The Rust side already validated this; a mismatch here is a bug.
func parseMirror(raw string) (mirror, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return mirror{}, fmt.Errorf("empty mirror")
	}
	if !strings.Contains(raw, "://") {
		raw = "https://" + raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return mirror{}, fmt.Errorf("mirror %q: %w", raw, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return mirror{}, fmt.Errorf("mirror %q: scheme must be http or https", raw)
	}
	if u.Host == "" {
		return mirror{}, fmt.Errorf("mirror %q: no host", raw)
	}
	if u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return mirror{}, fmt.Errorf("mirror %q: only scheme, host and path are allowed", raw)
	}
	return mirror{u.Scheme, strings.ToLower(u.Host), strings.TrimRight(u.Path, "/")}, nil
}

// setRegistryMirrors replaces the configuration with the decoded form of
// `{"<host>": ["<mirror>", ...]}`; "" or "{}" clears it. The backend
// applies the candidate first, so a failure leaves the old one in force.
func setRegistryMirrors(mirrorsJSON string) error {
	var raw map[string][]string
	if mirrorsJSON != "" {
		if err := json.Unmarshal([]byte(mirrorsJSON), &raw); err != nil {
			return fmt.Errorf("decode registry mirrors: %w", err)
		}
	}
	byHost := make(mirrorSet, len(raw))
	for host, list := range raw {
		key := canonicalRegistryHost(host)
		if key == "" || strings.ContainsAny(key, "/ ") {
			return fmt.Errorf("registry %q: not a host", host)
		}
		for _, entry := range list {
			m, err := parseMirror(entry)
			if err != nil {
				return fmt.Errorf("registry %q: %w", host, err)
			}
			byHost[key] = append(byHost[key], m)
		}
	}
	if err := applyRegistryMirrors(byHost); err != nil {
		return err
	}
	registryMirrors.mu.Lock()
	registryMirrors.byHost = byHost
	registryMirrors.mu.Unlock()
	return nil
}

// mirrorsFor returns the mirrors for the registry a reference names.
func mirrorsFor(host string) []mirror {
	registryMirrors.mu.RLock()
	defer registryMirrors.mu.RUnlock()
	return registryMirrors.byHost[canonicalRegistryHost(host)]
}

// hosts lists the mirrored registries, sorted.
func (s mirrorSet) hosts() []string {
	out := make([]string, 0, len(s))
	for host, list := range s {
		if len(list) > 0 {
			out = append(out, host)
		}
	}
	sort.Strings(out)
	return out
}

// hostsWithMirrors lists the configured mirrored registries, sorted.
func hostsWithMirrors() []string {
	registryMirrors.mu.RLock()
	defer registryMirrors.mu.RUnlock()
	return registryMirrors.byHost.hosts()
}

type noMirrorsKey struct{}

// withoutMirrors marks ctx so registry reads go straight to the registry.
// Signature artifacts use it: appendSignature rebuilds the signature
// manifest from what it reads, and a stale mirror would have it drop
// other signers' signatures.
func withoutMirrors(ctx context.Context) context.Context {
	return context.WithValue(ctx, noMirrorsKey{}, true)
}

func mirrorsDisabled(ctx context.Context) bool {
	v, _ := ctx.Value(noMirrorsKey{}).(bool)
	return v
}

// llmman_set_registry_mirrors configures the mirrors every later pull
// consults, replacing the previous configuration.
//
//export llmman_set_registry_mirrors
func llmman_set_registry_mirrors(cMirrorsJSON *C.char) *C.char {
	if err := setRegistryMirrors(C.GoString(cMirrorsJSON)); err != nil {
		return errResp(err)
	}
	return okResp("")
}
