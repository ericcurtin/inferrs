package main

import (
	"strings"
	"testing"
)

func TestParseMirrorAcceptsTheDocumentedShapesAndNothingElse(t *testing.T) {
	for raw, want := range map[string]mirror{
		"mirror.gcr.io":                      {scheme: "https", host: "mirror.gcr.io"},
		"https://Mirror.GCR.io/":             {scheme: "https", host: "mirror.gcr.io"},
		"http://registry-mirror.corp:5000":   {scheme: "http", host: "registry-mirror.corp:5000"},
		"https://proxy.corp/registry/":       {scheme: "https", host: "proxy.corp", path: "/registry"},
		" http://127.0.0.1:5000 ":            {scheme: "http", host: "127.0.0.1:5000"},
		"https://[::1]:5000":                 {scheme: "https", host: "[::1]:5000"},
		"https://mirror.example.com:443/v2/": {scheme: "https", host: "mirror.example.com:443", path: "/v2"},
	} {
		got, err := parseMirror(raw)
		if err != nil {
			t.Errorf("parseMirror(%q): %v", raw, err)
			continue
		}
		if got != want {
			t.Errorf("parseMirror(%q) = %+v, want %+v", raw, got, want)
		}
	}
	for _, bad := range []string{"", "   ", "ftp://mirror", "https://", "https://user:pw@mirror", "https://mirror?ns=x", "https://mirror#frag"} {
		if _, err := parseMirror(bad); err == nil {
			t.Errorf("parseMirror(%q) accepted", bad)
		}
	}
}

func TestSetRegistryMirrorsReplacesAndCanonicalizesHosts(t *testing.T) {
	// The podman backend writes the configuration out under the home
	// directory (see registries_podman.go); keep that out of the real one.
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Cleanup(func() { _ = setRegistryMirrors("") })

	if err := setRegistryMirrors(`{"index.docker.io": ["https://a", "b:5000"], "GHCR.io": ["http://g"]}`); err != nil {
		t.Fatalf("set: %v", err)
	}
	// Hub's names all reach the same list, in the configured order.
	for _, host := range []string{"docker.io", "index.docker.io", "registry-1.docker.io"} {
		got := mirrorsFor(host)
		if len(got) != 2 || got[0].String() != "https://a" || got[1].String() != "https://b:5000" {
			t.Errorf("mirrorsFor(%q) = %v", host, got)
		}
	}
	if got := mirrorsFor("ghcr.io"); len(got) != 1 || got[0].String() != "http://g" {
		t.Errorf("mirrorsFor(ghcr.io) = %v", got)
	}
	if got := mirrorsFor("quay.io"); got != nil {
		t.Errorf("mirrorsFor(quay.io) = %v, want none", got)
	}
	if got := hostsWithMirrors(); strings.Join(got, ",") != "docker.io,ghcr.io" {
		t.Errorf("hostsWithMirrors = %v", got)
	}

	// A second call replaces the first rather than adding to it.
	if err := setRegistryMirrors(`{"quay.io": ["q"]}`); err != nil {
		t.Fatalf("set again: %v", err)
	}
	if got := mirrorsFor("docker.io"); got != nil {
		t.Errorf("docker.io kept its mirrors across a replacement: %v", got)
	}
	if got := mirrorsFor("quay.io"); len(got) != 1 {
		t.Errorf("mirrorsFor(quay.io) = %v", got)
	}

	// Clearing.
	for _, empty := range []string{"", "{}"} {
		if err := setRegistryMirrors(empty); err != nil {
			t.Fatalf("clear with %q: %v", empty, err)
		}
		if got := hostsWithMirrors(); len(got) != 0 {
			t.Errorf("after clearing with %q: %v", empty, got)
		}
	}

	// A bad entry is refused whole, leaving the previous configuration.
	if err := setRegistryMirrors(`{"docker.io": ["ok"]}`); err != nil {
		t.Fatalf("set: %v", err)
	}
	for _, bad := range []string{`{"docker.io": ["ftp://x"]}`, `{"docker.io/ai": ["x"]}`, `not json`} {
		if err := setRegistryMirrors(bad); err == nil {
			t.Errorf("setRegistryMirrors(%s) accepted", bad)
		}
		if got := mirrorsFor("docker.io"); len(got) != 1 || got[0].String() != "https://ok" {
			t.Errorf("a refused configuration disturbed the previous one: %v", got)
		}
	}
}
