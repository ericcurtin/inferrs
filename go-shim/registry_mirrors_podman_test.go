//go:build podman

package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"go.podman.io/image/v5/docker/reference"
	"go.podman.io/image/v5/pkg/sysregistriesv2"
	"go.podman.io/image/v5/types"
)

func mustParseRef(t *testing.T, ref string) reference.Named {
	t.Helper()
	named, err := reference.ParseNormalizedNamed(ref)
	if err != nil {
		t.Fatal(err)
	}
	return named
}

// TestMirrorsReachPodmanThroughARegistriesConfDropIn proves the whole
// podman path: configured mirrors are written out as a drop-in that
// go.podman.io/image's own parser reads back as that registry's mirrors,
// alongside (not instead of) the main registries.conf, and only for the
// SystemContext registrySourceCtx hands out.
func TestMirrorsReachPodmanThroughARegistriesConfDropIn(t *testing.T) {
	// Keep the drop-in out of the real ~/.local/share/llmman.
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	// A main file of our own, with a registry of its own, standing in
	// for whatever ensureRegistriesConf (or the user) pointed the
	// variable at.
	mainConf := filepath.Join(home, "registries.conf")
	if err := os.WriteFile(mainConf, []byte(`unqualified-search-registries = []

[[registry]]
prefix = "quay.io"
location = "quay.io"

[[registry.mirror]]
location = "quay-cache.corp"

[[registry]]
prefix = "docker.io"
location = "hub-proxy.corp"
insecure = true
blocked = true

[[registry.mirror]]
location = "old-mirror.corp"
`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CONTAINERS_REGISTRIES_CONF", mainConf)
	t.Cleanup(func() { _ = setRegistryMirrors("") })

	if err := setRegistryMirrors(`{"index.docker.io": ["https://mirror.gcr.io", "http://cache.corp:5000/hub"]}`); err != nil {
		t.Fatalf("set: %v", err)
	}

	sys := registrySourceCtx(context.Background())
	if sys.SystemRegistriesConfPath != mainConf {
		t.Errorf("main file = %q, want %q", sys.SystemRegistriesConfPath, mainConf)
	}
	if !strings.HasPrefix(sys.SystemRegistriesConfDirPath, home) {
		t.Errorf("drop-in dir %q is not under the test home", sys.SystemRegistriesConfDirPath)
	}

	reg, err := sysregistriesv2.FindRegistry(sys, "docker.io/ai/smollm2:latest")
	if err != nil {
		t.Fatalf("FindRegistry: %v", err)
	}
	if reg == nil {
		t.Fatalf("docker.io has no [[registry]] entry; drop-in dir: %v", listDir(t, sys.SystemRegistriesConfDirPath))
	}
	if len(reg.Mirrors) != 2 {
		t.Fatalf("docker.io mirrors = %+v, want llmman.conf's 2, not the main file's", reg.Mirrors)
	}
	// Everything but the mirror list is the main file's.
	if reg.Location != "hub-proxy.corp" || !reg.Insecure || !reg.Blocked {
		t.Errorf("main file's docker.io settings were not carried over: %+v", reg)
	}
	if reg.Mirrors[0].Location != "mirror.gcr.io" || reg.Mirrors[0].Insecure {
		t.Errorf("first mirror = %+v", reg.Mirrors[0])
	}
	if reg.Mirrors[1].Location != "cache.corp:5000/hub" || !reg.Mirrors[1].Insecure {
		t.Errorf("second (http) mirror = %+v, want insecure at cache.corp:5000/hub", reg.Mirrors[1])
	}
	// The order a pull tries them in: mirrors, then the registry.
	sources, err := reg.PullSourcesFromReference(mustParseRef(t, "docker.io/ai/smollm2:latest"))
	if err != nil {
		t.Fatalf("PullSourcesFromReference: %v", err)
	}
	var order []string
	for _, s := range sources {
		order = append(order, s.Endpoint.Location)
	}
	if got := strings.Join(order, ","); got != "mirror.gcr.io,cache.corp:5000/hub,hub-proxy.corp" {
		t.Errorf("pull order = %s", got)
	}

	// The main file's own registry survives the drop-in.
	reg, err = sysregistriesv2.FindRegistry(sys, "quay.io/org/x:1")
	if err != nil || reg == nil || len(reg.Mirrors) != 1 || reg.Mirrors[0].Location != "quay-cache.corp" {
		t.Errorf("quay.io from the main file = %+v (%v)", reg, err)
	}

	// Write-side calls use the empty context and see only the main file;
	// so does anything under withoutMirrors (signatures).
	for _, c := range []*types.SystemContext{{}, registrySourceCtx(withoutMirrors(context.Background()))} {
		reg, err = sysregistriesv2.FindRegistry(c, "docker.io/ai/smollm2:latest")
		if err != nil {
			t.Fatal(err)
		}
		if reg == nil || len(reg.Mirrors) != 1 || reg.Mirrors[0].Location != "old-mirror.corp" {
			t.Errorf("a mirror-free context picked llmman's mirrors up: %+v", reg)
		}
	}

	// The same configuration lands in the same directory; clearing hands
	// out the empty context again.
	dir := sys.SystemRegistriesConfDirPath
	if err := setRegistryMirrors(`{"docker.io": ["https://mirror.gcr.io", "http://cache.corp:5000/hub"]}`); err != nil {
		t.Fatal(err)
	}
	if got := registrySourceCtx(context.Background()).SystemRegistriesConfDirPath; got != dir {
		t.Errorf("same configuration moved from %s to %s", dir, got)
	}
	if err := setRegistryMirrors(""); err != nil {
		t.Fatal(err)
	}
	if got := registrySourceCtx(context.Background()); got.SystemRegistriesConfDirPath != "" || got.SystemRegistriesConfPath != "" {
		t.Errorf("context after clearing = %+v", got)
	}
}

func TestMirrorsRegistriesConfIsValidV2TOML(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Cleanup(func() { _ = setRegistryMirrors("") })
	if err := setRegistryMirrors(`{"ghcr.io": ["ghcr-cache.corp"], "docker.io": ["http://hub-cache.corp"]}`); err != nil {
		t.Fatal(err)
	}
	got := mirrorsRegistriesConf(registryMirrors.byHost, &types.SystemContext{SystemRegistriesConfPath: filepath.Join(t.TempDir(), "none.conf")})
	for _, want := range []string{
		"[[registry]]\nprefix = \"docker.io\"\nlocation = \"docker.io\"\n",
		"[[registry.mirror]]\nlocation = \"hub-cache.corp\"\ninsecure = true\n",
		"[[registry]]\nprefix = \"ghcr.io\"\nlocation = \"ghcr.io\"\n",
		"[[registry.mirror]]\nlocation = \"ghcr-cache.corp\"\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	if strings.Index(got, "docker.io") > strings.Index(got, "ghcr.io") {
		t.Errorf("hosts are not in sorted order:\n%s", got)
	}
}

func listDir(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return []string{err.Error()}
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names
}
