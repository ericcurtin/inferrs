//go:build !cshared

// inferrs-oci-models — OCI model helper for inferrs (standalone CLI).
//
// Uses Docker Model Runner's distribution library (containerd transport)
// to pull models from OCI registries.  Shares the same store as DMR at
// ~/.docker/models so models already pulled by DMR are reused and vice versa.
//
// Commands:
//
//	inferrs-oci-models pull  <reference>   Pull a model and print the bundle path.
//	inferrs-oci-models bundle <reference>  Print the bundle path for an already-pulled model.
//	inferrs-oci-models list                List all models in the store (tag\tid).
package main

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/docker/model-runner/cmd/cli/desktop"
)

func cmdPull(ref string) error {
	client, err := newClient()
	if err != nil {
		return err
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
		// Use DMR's exact progress rendering (Docker-style multi-line bars).
		msg, progress, err := desktop.DisplayProgress(pr, &stderrPrinter{})
		done <- pullResult{msg, progress, err}
	}()

	// Pull the model.
	pullErr := client.PullModel(context.Background(), ref, pw)
	pw.Close()

	// Wait for progress rendering to finish.
	result := <-done
	if pullErr != nil {
		return fmt.Errorf("pull: %w", pullErr)
	}
	if result.err != nil {
		return fmt.Errorf("progress: %w", result.err)
	}

	// Print final message from progress stream (e.g. "Model pulled successfully").
	if result.msg != "" {
		fmt.Fprintf(os.Stderr, "%s\n", result.msg)
	}

	// Ensure the bundle is created and print its root directory.
	bundle, err := client.GetBundle(ref)
	if err != nil {
		return fmt.Errorf("bundle: %w", err)
	}

	// stdout: the bundle path — this is what the Rust caller reads.
	fmt.Println(bundle.RootDir())
	return nil
}

func cmdBundle(ref string) error {
	client, err := newClient()
	if err != nil {
		return err
	}

	bundle, err := client.GetBundle(ref)
	if err != nil {
		return fmt.Errorf("bundle: %w", err)
	}

	fmt.Println(bundle.RootDir())
	return nil
}

func cmdList() error {
	client, err := newClient()
	if err != nil {
		return err
	}

	models, err := client.ListModels()
	if err != nil {
		return fmt.Errorf("list: %w", err)
	}

	for _, m := range models {
		id, err := m.ID()
		if err != nil {
			continue
		}
		for _, tag := range m.Tags() {
			fmt.Printf("%s\t%s\n", tag, id)
		}
	}
	return nil
}

func usage() {
	fmt.Fprintf(os.Stderr, "Usage: inferrs-oci-models <command> [args]\n\n")
	fmt.Fprintf(os.Stderr, "Commands:\n")
	fmt.Fprintf(os.Stderr, "  pull  <reference>   Pull a model and print bundle path\n")
	fmt.Fprintf(os.Stderr, "  bundle <reference>  Print bundle path for existing model\n")
	fmt.Fprintf(os.Stderr, "  list                List all models in the store\n")
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	var err error
	switch os.Args[1] {
	case "pull":
		if len(os.Args) < 3 {
			fmt.Fprintf(os.Stderr, "Usage: inferrs-oci-models pull <reference>\n")
			os.Exit(1)
		}
		err = cmdPull(os.Args[2])
	case "bundle":
		if len(os.Args) < 3 {
			fmt.Fprintf(os.Stderr, "Usage: inferrs-oci-models bundle <reference>\n")
			os.Exit(1)
		}
		err = cmdBundle(os.Args[2])
	case "list":
		err = cmdList()
	default:
		usage()
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
