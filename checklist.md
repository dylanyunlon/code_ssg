tree
.
├── AGENTS.md
├── bunfig.toml
├── bun.lock
├── CONTRIBUTING.md
├── flake.lock
├── flake.nix
├── github
│   ├── action.yml
│   ├── bun.lock
│   ├── index.ts
│   ├── package.json
│   ├── README.md
│   ├── script
│   │   ├── publish
│   │   └── release
│   ├── sst-env.d.ts
│   └── tsconfig.json
├── infra
│   ├── app.ts
│   ├── console.ts
│   ├── enterprise.ts
│   ├── secret.ts
│   └── stage.ts
├── install
├── LICENSE
├── nix
│   ├── desktop.nix
│   ├── hashes.json
│   ├── node_modules.nix
│   ├── opencode.nix
│   └── scripts
│       ├── canonicalize-node-modules.ts
│       └── normalize-bun-binaries.ts
├── package.json
├── packages
│   ├── app
│   │   ├── AGENTS.md
│   │   ├── bunfig.toml
│   │   ├── e2e
│   │   │   ├── actions.ts
│   │   │   ├── AGENTS.md
│   │   │   ├── app
│   │   │   │   ├── home.spec.ts
│   │   │   │   ├── navigation.spec.ts
│   │   │   │   ├── palette.spec.ts
│   │   │   │   ├── server-default.spec.ts
│   │   │   │   ├── session.spec.ts
│   │   │   │   └── titlebar-history.spec.ts
│   │   │   ├── commands
│   │   │   │   ├── input-focus.spec.ts
│   │   │   │   ├── panels.spec.ts
│   │   │   │   └── tab-close.spec.ts
│   │   │   ├── files
│   │   │   │   ├── file-open.spec.ts
│   │   │   │   ├── file-tree.spec.ts
│   │   │   │   └── file-viewer.spec.ts
│   │   │   ├── fixtures.ts
│   │   │   ├── models
│   │   │   │   ├── model-picker.spec.ts
│   │   │   │   └── models-visibility.spec.ts
│   │   │   ├── projects
│   │   │   │   ├── project-edit.spec.ts
│   │   │   │   ├── projects-close.spec.ts
│   │   │   │   ├── projects-switch.spec.ts
│   │   │   │   ├── workspace-new-session.spec.ts
│   │   │   │   └── workspaces.spec.ts
│   │   │   ├── prompt
│   │   │   │   ├── context.spec.ts
│   │   │   │   ├── prompt-async.spec.ts
│   │   │   │   ├── prompt-drop-file.spec.ts
│   │   │   │   ├── prompt-drop-file-uri.spec.ts
│   │   │   │   ├── prompt-mention.spec.ts
│   │   │   │   ├── prompt-multiline.spec.ts
│   │   │   │   ├── prompt-slash-open.spec.ts
│   │   │   │   ├── prompt-slash-terminal.spec.ts
│   │   │   │   └── prompt.spec.ts
│   │   │   ├── selectors.ts
│   │   │   ├── session
│   │   │   │   ├── session-composer-dock.spec.ts
│   │   │   │   ├── session.spec.ts
│   │   │   │   └── session-undo-redo.spec.ts
│   │   │   ├── settings
│   │   │   │   ├── settings-keybinds.spec.ts
│   │   │   │   ├── settings-models.spec.ts
│   │   │   │   ├── settings-providers.spec.ts
│   │   │   │   └── settings.spec.ts
│   │   │   ├── sidebar
│   │   │   │   ├── sidebar-popover-actions.spec.ts
│   │   │   │   ├── sidebar-session-links.spec.ts
│   │   │   │   └── sidebar.spec.ts
│   │   │   ├── status
│   │   │   │   └── status-popover.spec.ts
│   │   │   ├── terminal
│   │   │   │   ├── terminal-init.spec.ts
│   │   │   │   └── terminal.spec.ts
│   │   │   ├── thinking-level.spec.ts
│   │   │   ├── tsconfig.json
│   │   │   └── utils.ts
│   │   ├── happydom.ts
│   │   ├── index.html
│   │   ├── package.json
│   │   ├── playwright.config.ts
│   │   ├── public
│   │   │   ├── apple-touch-icon.png -> ../../ui/src/assets/favicon/apple-touch-icon.png
│   │   │   ├── apple-touch-icon-v3.png -> ../../ui/src/assets/favicon/apple-touch-icon-v3.png
│   │   │   ├── favicon-96x96.png -> ../../ui/src/assets/favicon/favicon-96x96.png
│   │   │   ├── favicon-96x96-v3.png -> ../../ui/src/assets/favicon/favicon-96x96-v3.png
│   │   │   ├── favicon.ico -> ../../ui/src/assets/favicon/favicon.ico
│   │   │   ├── favicon.svg -> ../../ui/src/assets/favicon/favicon.svg
│   │   │   ├── favicon-v3.ico -> ../../ui/src/assets/favicon/favicon-v3.ico
│   │   │   ├── favicon-v3.svg -> ../../ui/src/assets/favicon/favicon-v3.svg
│   │   │   ├── _headers
│   │   │   ├── oc-theme-preload.js
│   │   │   ├── site.webmanifest -> ../../ui/src/assets/favicon/site.webmanifest
│   │   │   ├── social-share.png -> ../../ui/src/assets/images/social-share.png
│   │   │   ├── social-share-zen.png -> ../../ui/src/assets/images/social-share-zen.png
│   │   │   ├── web-app-manifest-192x192.png -> ../../ui/src/assets/favicon/web-app-manifest-192x192.png
│   │   │   └── web-app-manifest-512x512.png -> ../../ui/src/assets/favicon/web-app-manifest-512x512.png
│   │   ├── README.md
│   │   ├── script
│   │   │   └── e2e-local.ts
│   │   ├── src
│   │   │   ├── addons
│   │   │   │   ├── serialize.test.ts
│   │   │   │   └── serialize.ts
│   │   │   ├── app.tsx
│   │   │   ├── components
│   │   │   │   ├── dialog-connect-provider.tsx
│   │   │   │   ├── dialog-custom-provider.tsx
│   │   │   │   ├── dialog-edit-project.tsx
│   │   │   │   ├── dialog-fork.tsx
│   │   │   │   ├── dialog-manage-models.tsx
│   │   │   │   ├── dialog-release-notes.tsx
│   │   │   │   ├── dialog-select-directory.tsx
│   │   │   │   ├── dialog-select-file.tsx
│   │   │   │   ├── dialog-select-mcp.tsx
│   │   │   │   ├── dialog-select-model.tsx
│   │   │   │   ├── dialog-select-model-unpaid.tsx
│   │   │   │   ├── dialog-select-provider.tsx
│   │   │   │   ├── dialog-select-server.tsx
│   │   │   │   ├── dialog-settings.tsx
│   │   │   │   ├── file-tree.test.ts
│   │   │   │   ├── file-tree.tsx
│   │   │   │   ├── link.tsx
│   │   │   │   ├── model-tooltip.tsx
│   │   │   │   ├── prompt-input
│   │   │   │   │   ├── attachments.ts
│   │   │   │   │   ├── build-request-parts.test.ts
│   │   │   │   │   ├── build-request-parts.ts
│   │   │   │   │   ├── context-items.tsx
│   │   │   │   │   ├── drag-overlay.tsx
│   │   │   │   │   ├── editor-dom.test.ts
│   │   │   │   │   ├── editor-dom.ts
│   │   │   │   │   ├── history.test.ts
│   │   │   │   │   ├── history.ts
│   │   │   │   │   ├── image-attachments.tsx
│   │   │   │   │   ├── placeholder.test.ts
│   │   │   │   │   ├── placeholder.ts
│   │   │   │   │   ├── slash-popover.tsx
│   │   │   │   │   ├── submit.test.ts
│   │   │   │   │   └── submit.ts
│   │   │   │   ├── prompt-input.tsx
│   │   │   │   ├── server
│   │   │   │   │   └── server-row.tsx
│   │   │   │   ├── session
│   │   │   │   │   ├── index.ts
│   │   │   │   │   ├── session-context-breakdown.test.ts
│   │   │   │   │   ├── session-context-breakdown.ts
│   │   │   │   │   ├── session-context-format.ts
│   │   │   │   │   ├── session-context-metrics.test.ts
│   │   │   │   │   ├── session-context-metrics.ts
│   │   │   │   │   ├── session-context-tab.tsx
│   │   │   │   │   ├── session-header.tsx
│   │   │   │   │   ├── session-new-view.tsx
│   │   │   │   │   ├── session-sortable-tab.tsx
│   │   │   │   │   └── session-sortable-terminal-tab.tsx
│   │   │   │   ├── session-context-usage.tsx
│   │   │   │   ├── settings-agents.tsx
│   │   │   │   ├── settings-commands.tsx
│   │   │   │   ├── settings-general.tsx
│   │   │   │   ├── settings-keybinds.tsx
│   │   │   │   ├── settings-mcp.tsx
│   │   │   │   ├── settings-models.tsx
│   │   │   │   ├── settings-permissions.tsx
│   │   │   │   ├── settings-providers.tsx
│   │   │   │   ├── status-popover.tsx
│   │   │   │   ├── terminal.tsx
│   │   │   │   ├── titlebar-history.test.ts
│   │   │   │   ├── titlebar-history.ts
│   │   │   │   └── titlebar.tsx
│   │   │   ├── context
│   │   │   │   ├── command-keybind.test.ts
│   │   │   │   ├── command.test.ts
│   │   │   │   ├── command.tsx
│   │   │   │   ├── comments.test.ts
│   │   │   │   ├── comments.tsx
│   │   │   │   ├── file
│   │   │   │   │   ├── content-cache.ts
│   │   │   │   │   ├── path.test.ts
│   │   │   │   │   ├── path.ts
│   │   │   │   │   ├── tree-store.ts
│   │   │   │   │   ├── types.ts
│   │   │   │   │   ├── view-cache.ts
│   │   │   │   │   ├── watcher.test.ts
│   │   │   │   │   └── watcher.ts
│   │   │   │   ├── file-content-eviction-accounting.test.ts
│   │   │   │   ├── file.tsx
│   │   │   │   ├── global-sdk.tsx
│   │   │   │   ├── global-sync
│   │   │   │   │   ├── bootstrap.ts
│   │   │   │   │   ├── child-store.test.ts
│   │   │   │   │   ├── child-store.ts
│   │   │   │   │   ├── event-reducer.test.ts
│   │   │   │   │   ├── event-reducer.ts
│   │   │   │   │   ├── eviction.ts
│   │   │   │   │   ├── queue.ts
│   │   │   │   │   ├── session-load.ts
│   │   │   │   │   ├── session-trim.test.ts
│   │   │   │   │   ├── session-trim.ts
│   │   │   │   │   ├── types.ts
│   │   │   │   │   └── utils.ts
│   │   │   │   ├── global-sync.test.ts
│   │   │   │   ├── global-sync.tsx
│   │   │   │   ├── highlights.tsx
│   │   │   │   ├── language.tsx
│   │   │   │   ├── layout-scroll.test.ts
│   │   │   │   ├── layout-scroll.ts
│   │   │   │   ├── layout.test.ts
│   │   │   │   ├── layout.tsx
│   │   │   │   ├── local.tsx
│   │   │   │   ├── models.tsx
│   │   │   │   ├── model-variant.test.ts
│   │   │   │   ├── model-variant.ts
│   │   │   │   ├── notification-index.ts
│   │   │   │   ├── notification.test.ts
│   │   │   │   ├── notification.tsx
│   │   │   │   ├── permission.tsx
│   │   │   │   ├── platform.tsx
│   │   │   │   ├── prompt.tsx
│   │   │   │   ├── sdk.tsx
│   │   │   │   ├── server.tsx
│   │   │   │   ├── settings.tsx
│   │   │   │   ├── sync-optimistic.test.ts
│   │   │   │   ├── sync.tsx
│   │   │   │   ├── terminal.test.ts
│   │   │   │   └── terminal.tsx
│   │   │   ├── custom-elements.d.ts -> ../../ui/src/custom-elements.d.ts
│   │   │   ├── entry.tsx
│   │   │   ├── env.d.ts
│   │   │   ├── hooks
│   │   │   │   └── use-providers.ts
│   │   │   ├── i18n
│   │   │   │   ├── ar.ts
│   │   │   │   ├── br.ts
│   │   │   │   ├── bs.ts
│   │   │   │   ├── da.ts
│   │   │   │   ├── de.ts
│   │   │   │   ├── en.ts
│   │   │   │   ├── es.ts
│   │   │   │   ├── fr.ts
│   │   │   │   ├── ja.ts
│   │   │   │   ├── ko.ts
│   │   │   │   ├── no.ts
│   │   │   │   ├── parity.test.ts
│   │   │   │   ├── pl.ts
│   │   │   │   ├── ru.ts
│   │   │   │   ├── th.ts
│   │   │   │   ├── zh.ts
│   │   │   │   └── zht.ts
│   │   │   ├── index.css
│   │   │   ├── index.ts
│   │   │   ├── pages
│   │   │   │   ├── directory-layout.tsx
│   │   │   │   ├── error.tsx
│   │   │   │   ├── home.tsx
│   │   │   │   ├── layout
│   │   │   │   │   ├── deep-links.ts
│   │   │   │   │   ├── helpers.test.ts
│   │   │   │   │   ├── helpers.ts
│   │   │   │   │   ├── inline-editor.tsx
│   │   │   │   │   ├── sidebar-items.tsx
│   │   │   │   │   ├── sidebar-project-helpers.test.ts
│   │   │   │   │   ├── sidebar-project-helpers.ts
│   │   │   │   │   ├── sidebar-project.tsx
│   │   │   │   │   ├── sidebar-shell-helpers.ts
│   │   │   │   │   ├── sidebar-shell.test.ts
│   │   │   │   │   ├── sidebar-shell.tsx
│   │   │   │   │   ├── sidebar-workspace-helpers.ts
│   │   │   │   │   ├── sidebar-workspace.test.ts
│   │   │   │   │   └── sidebar-workspace.tsx
│   │   │   │   ├── layout.tsx
│   │   │   │   ├── session
│   │   │   │   │   ├── composer
│   │   │   │   │   │   ├── index.ts
│   │   │   │   │   │   ├── session-composer-region.tsx
│   │   │   │   │   │   ├── session-composer-state.ts
│   │   │   │   │   │   ├── session-permission-dock.tsx
│   │   │   │   │   │   ├── session-question-dock.tsx
│   │   │   │   │   │   └── session-todo-dock.tsx
│   │   │   │   │   ├── file-tab-scroll.test.ts
│   │   │   │   │   ├── file-tab-scroll.ts
│   │   │   │   │   ├── file-tabs.tsx
│   │   │   │   │   ├── handoff.ts
│   │   │   │   │   ├── helpers.test.ts
│   │   │   │   │   ├── helpers.ts
│   │   │   │   │   ├── message-gesture.test.ts
│   │   │   │   │   ├── message-gesture.ts
│   │   │   │   │   ├── message-timeline.tsx
│   │   │   │   │   ├── review-tab.tsx
│   │   │   │   │   ├── scroll-spy.test.ts
│   │   │   │   │   ├── scroll-spy.ts
│   │   │   │   │   ├── session-command-helpers.ts
│   │   │   │   │   ├── session-mobile-tabs.tsx
│   │   │   │   │   ├── session-prompt-dock.test.ts
│   │   │   │   │   ├── session-prompt-helpers.ts
│   │   │   │   │   ├── session-side-panel.tsx
│   │   │   │   │   ├── terminal-label.ts
│   │   │   │   │   ├── terminal-panel.test.ts
│   │   │   │   │   ├── terminal-panel.tsx
│   │   │   │   │   ├── use-session-commands.test.ts
│   │   │   │   │   ├── use-session-commands.tsx
│   │   │   │   │   ├── use-session-hash-scroll.test.ts
│   │   │   │   │   └── use-session-hash-scroll.ts
│   │   │   │   └── session.tsx
│   │   │   ├── sst-env.d.ts
│   │   │   └── utils
│   │   │       ├── agent.ts
│   │   │       ├── aim.ts
│   │   │       ├── base64.ts
│   │   │       ├── dom.ts
│   │   │       ├── id.ts
│   │   │       ├── index.ts
│   │   │       ├── notification-click.test.ts
│   │   │       ├── notification-click.ts
│   │   │       ├── persist.test.ts
│   │   │       ├── persist.ts
│   │   │       ├── prompt.ts
│   │   │       ├── runtime-adapters.test.ts
│   │   │       ├── runtime-adapters.ts
│   │   │       ├── same.ts
│   │   │       ├── scoped-cache.test.ts
│   │   │       ├── scoped-cache.ts
│   │   │       ├── server-errors.test.ts
│   │   │       ├── server-errors.ts
│   │   │       ├── server-health.test.ts
│   │   │       ├── server-health.ts
│   │   │       ├── server.ts
│   │   │       ├── solid-dnd.tsx
│   │   │       ├── sound.ts
│   │   │       ├── speech.ts
│   │   │       ├── terminal-writer.test.ts
│   │   │       ├── terminal-writer.ts
│   │   │       ├── time.ts
│   │   │       ├── uuid.test.ts
│   │   │       ├── uuid.ts
│   │   │       ├── worktree.test.ts
│   │   │       └── worktree.ts
│   │   ├── sst-env.d.ts
│   │   ├── tsconfig.json
│   │   ├── vite.config.ts
│   │   └── vite.js
│   ├── console
│   │   ├── app
│   │   │   ├── package.json
│   │   │   ├── public
│   │   │   │   ├── apple-touch-icon.png -> ../../../ui/src/assets/favicon/apple-touch-icon.png
│   │   │   │   ├── apple-touch-icon-v3.png -> ../../../ui/src/assets/favicon/apple-touch-icon-v3.png
│   │   │   │   ├── email -> ../../mail/emails/templates/static
│   │   │   │   ├── favicon-96x96.png -> ../../../ui/src/assets/favicon/favicon-96x96.png
│   │   │   │   ├── favicon-96x96-v3.png -> ../../../ui/src/assets/favicon/favicon-96x96-v3.png
│   │   │   │   ├── favicon.ico -> ../../../ui/src/assets/favicon/favicon.ico
│   │   │   │   ├── favicon.svg -> ../../../ui/src/assets/favicon/favicon.svg
│   │   │   │   ├── favicon-v3.ico -> ../../../ui/src/assets/favicon/favicon-v3.ico
│   │   │   │   ├── favicon-v3.svg -> ../../../ui/src/assets/favicon/favicon-v3.svg
│   │   │   │   ├── opencode-brand-assets.zip
│   │   │   │   ├── robots.txt
│   │   │   │   ├── site.webmanifest -> ../../../ui/src/assets/favicon/site.webmanifest
│   │   │   │   ├── social-share-black.png -> ../../../ui/src/assets/images/social-share-black.png
│   │   │   │   ├── social-share.png -> ../../../ui/src/assets/images/social-share.png
│   │   │   │   ├── social-share-zen.png -> ../../../ui/src/assets/images/social-share-zen.png
│   │   │   │   ├── theme.json
│   │   │   │   ├── web-app-manifest-192x192.png -> ../../../ui/src/assets/favicon/web-app-manifest-192x192.png
│   │   │   │   └── web-app-manifest-512x512.png -> ../../../ui/src/assets/favicon/web-app-manifest-512x512.png
│   │   │   ├── README.md
│   │   │   ├── script
│   │   │   │   └── generate-sitemap.ts
│   │   │   ├── src
│   │   │   │   ├── app.css
│   │   │   │   ├── app.tsx
│   │   │   │   ├── asset
│   │   │   │   │   ├── black
│   │   │   │   │   │   └── hero.png
│   │   │   │   │   ├── brand
│   │   │   │   │   │   ├── opencode-brand-assets.zip
│   │   │   │   │   │   ├── opencode-logo-dark.png
│   │   │   │   │   │   ├── opencode-logo-dark-square.png
│   │   │   │   │   │   ├── opencode-logo-dark-square.svg
│   │   │   │   │   │   ├── opencode-logo-dark.svg
│   │   │   │   │   │   ├── opencode-logo-light.png
│   │   │   │   │   │   ├── opencode-logo-light-square.png
│   │   │   │   │   │   ├── opencode-logo-light-square.svg
│   │   │   │   │   │   ├── opencode-logo-light.svg
│   │   │   │   │   │   ├── opencode-wordmark-dark.png
│   │   │   │   │   │   ├── opencode-wordmark-dark.svg
│   │   │   │   │   │   ├── opencode-wordmark-light.png
│   │   │   │   │   │   ├── opencode-wordmark-light.svg
│   │   │   │   │   │   ├── opencode-wordmark-simple-dark.png
│   │   │   │   │   │   ├── opencode-wordmark-simple-dark.svg
│   │   │   │   │   │   ├── opencode-wordmark-simple-light.png
│   │   │   │   │   │   ├── opencode-wordmark-simple-light.svg
│   │   │   │   │   │   ├── preview-opencode-dark.png
│   │   │   │   │   │   ├── preview-opencode-logo-dark.png
│   │   │   │   │   │   ├── preview-opencode-logo-dark-square.png
│   │   │   │   │   │   ├── preview-opencode-logo-light.png
│   │   │   │   │   │   ├── preview-opencode-logo-light-square.png
│   │   │   │   │   │   ├── preview-opencode-wordmark-dark.png
│   │   │   │   │   │   ├── preview-opencode-wordmark-light.png
│   │   │   │   │   │   ├── preview-opencode-wordmark-simple-dark.png
│   │   │   │   │   │   └── preview-opencode-wordmark-simple-light.png
│   │   │   │   │   ├── lander
│   │   │   │   │   │   ├── avatar-adam.png
│   │   │   │   │   │   ├── avatar-david.png
│   │   │   │   │   │   ├── avatar-dax.png
│   │   │   │   │   │   ├── avatar-frank.png
│   │   │   │   │   │   ├── avatar-jay.png
│   │   │   │   │   │   ├── brand-assets-dark.svg
│   │   │   │   │   │   ├── brand-assets-light.svg
│   │   │   │   │   │   ├── brand.png
│   │   │   │   │   │   ├── check.svg
│   │   │   │   │   │   ├── copy.svg
│   │   │   │   │   │   ├── desktop-app-icon.png
│   │   │   │   │   │   ├── dock.png
│   │   │   │   │   │   ├── logo-dark.svg
│   │   │   │   │   │   ├── logo-light.svg
│   │   │   │   │   │   ├── opencode-comparison-min.mp4
│   │   │   │   │   │   ├── opencode-comparison-poster.png
│   │   │   │   │   │   ├── opencode-desktop-icon.png
│   │   │   │   │   │   ├── opencode-logo-dark.svg
│   │   │   │   │   │   ├── opencode-logo-light.svg
│   │   │   │   │   │   ├── opencode-min.mp4
│   │   │   │   │   │   ├── opencode-poster.png
│   │   │   │   │   │   ├── opencode-wordmark-dark.svg
│   │   │   │   │   │   ├── opencode-wordmark-light.svg
│   │   │   │   │   │   ├── screenshot-github.png
│   │   │   │   │   │   ├── screenshot.png
│   │   │   │   │   │   ├── screenshot-splash.png
│   │   │   │   │   │   ├── screenshot-vscode.png
│   │   │   │   │   │   ├── wordmark-dark.svg
│   │   │   │   │   │   └── wordmark-light.svg
│   │   │   │   │   ├── logo-ornate-dark.svg
│   │   │   │   │   ├── logo-ornate-light.svg
│   │   │   │   │   ├── logo.svg
│   │   │   │   │   ├── zen-ornate-dark.svg
│   │   │   │   │   └── zen-ornate-light.svg
│   │   │   │   ├── component
│   │   │   │   │   ├── dropdown.css
│   │   │   │   │   ├── dropdown.tsx
│   │   │   │   │   ├── email-signup.tsx
│   │   │   │   │   ├── faq.tsx
│   │   │   │   │   ├── footer.tsx
│   │   │   │   │   ├── header-context-menu.css
│   │   │   │   │   ├── header.tsx
│   │   │   │   │   ├── icon.tsx
│   │   │   │   │   ├── language-picker.css
│   │   │   │   │   ├── language-picker.tsx
│   │   │   │   │   ├── legal.tsx
│   │   │   │   │   ├── locale-links.tsx
│   │   │   │   │   ├── modal.css
│   │   │   │   │   ├── modal.tsx
│   │   │   │   │   ├── spotlight.css
│   │   │   │   │   └── spotlight.tsx
│   │   │   │   ├── config.ts
│   │   │   │   ├── context
│   │   │   │   │   ├── auth.session.ts
│   │   │   │   │   ├── auth.ts
│   │   │   │   │   ├── auth.withActor.ts
│   │   │   │   │   ├── i18n.tsx
│   │   │   │   │   └── language.tsx
│   │   │   │   ├── entry-client.tsx
│   │   │   │   ├── entry-server.tsx
│   │   │   │   ├── global.d.ts
│   │   │   │   ├── i18n
│   │   │   │   │   ├── ar.ts
│   │   │   │   │   ├── br.ts
│   │   │   │   │   ├── da.ts
│   │   │   │   │   ├── de.ts
│   │   │   │   │   ├── en.ts
│   │   │   │   │   ├── es.ts
│   │   │   │   │   ├── fr.ts
│   │   │   │   │   ├── index.ts
│   │   │   │   │   ├── it.ts
│   │   │   │   │   ├── ja.ts
│   │   │   │   │   ├── ko.ts
│   │   │   │   │   ├── no.ts
│   │   │   │   │   ├── pl.ts
│   │   │   │   │   ├── ru.ts
│   │   │   │   │   ├── th.ts
│   │   │   │   │   ├── tr.ts
│   │   │   │   │   ├── zh.ts
│   │   │   │   │   └── zht.ts
│   │   │   │   ├── lib
│   │   │   │   │   ├── changelog.ts
│   │   │   │   │   ├── form-error.ts
│   │   │   │   │   ├── github.ts
│   │   │   │   │   └── language.ts
│   │   │   │   ├── middleware.ts
│   │   │   │   ├── routes
│   │   │   │   │   ├── [...404].css
│   │   │   │   │   ├── [...404].tsx
│   │   │   │   │   ├── api
│   │   │   │   │   │   └── enterprise.ts
│   │   │   │   │   ├── auth
│   │   │   │   │   │   ├── authorize.ts
│   │   │   │   │   │   ├── [...callback].ts
│   │   │   │   │   │   ├── index.ts
│   │   │   │   │   │   ├── logout.ts
│   │   │   │   │   │   └── status.ts
│   │   │   │   │   ├── bench
│   │   │   │   │   │   ├── [id].tsx
│   │   │   │   │   │   ├── index.tsx
│   │   │   │   │   │   └── submission.ts
│   │   │   │   │   ├── black
│   │   │   │   │   │   ├── common.tsx
│   │   │   │   │   │   ├── index.tsx
│   │   │   │   │   │   ├── _subscribe
│   │   │   │   │   │   │   └── [plan].tsx
│   │   │   │   │   │   ├── workspace.css
│   │   │   │   │   │   └── workspace.tsx
│   │   │   │   │   ├── black.css
│   │   │   │   │   ├── black.tsx
│   │   │   │   │   ├── brand
│   │   │   │   │   │   ├── index.css
│   │   │   │   │   │   └── index.tsx
│   │   │   │   │   ├── changelog
│   │   │   │   │   │   ├── index.css
│   │   │   │   │   │   └── index.tsx
│   │   │   │   │   ├── changelog.json.ts
│   │   │   │   │   ├── debug
│   │   │   │   │   │   └── index.ts
│   │   │   │   │   ├── desktop-feedback.ts
│   │   │   │   │   ├── discord.ts
│   │   │   │   │   ├── docs
│   │   │   │   │   │   ├── index.ts
│   │   │   │   │   │   └── [...path].ts
│   │   │   │   │   ├── download
│   │   │   │   │   │   ├── [channel]
│   │   │   │   │   │   │   └── [platform].ts
│   │   │   │   │   │   ├── index.css
│   │   │   │   │   │   ├── index.tsx
│   │   │   │   │   │   └── types.ts
│   │   │   │   │   ├── enterprise
│   │   │   │   │   │   ├── index.css
│   │   │   │   │   │   └── index.tsx
│   │   │   │   │   ├── index.css
│   │   │   │   │   ├── index.tsx
│   │   │   │   │   ├── legal
│   │   │   │   │   │   ├── privacy-policy
│   │   │   │   │   │   │   ├── index.css
│   │   │   │   │   │   │   └── index.tsx
│   │   │   │   │   │   └── terms-of-service
│   │   │   │   │   │       ├── index.css
│   │   │   │   │   │       └── index.tsx
│   │   │   │   │   ├── openapi.json.ts
│   │   │   │   │   ├── s
│   │   │   │   │   │   └── [id].ts
│   │   │   │   │   ├── stripe
│   │   │   │   │   │   └── webhook.ts
│   │   │   │   │   ├── t
│   │   │   │   │   │   └── [...path].tsx
│   │   │   │   │   ├── temp.tsx
│   │   │   │   │   ├── user-menu.css
│   │   │   │   │   ├── user-menu.tsx
│   │   │   │   │   ├── workspace
│   │   │   │   │   │   ├── common.tsx
│   │   │   │   │   │   ├── [id]
│   │   │   │   │   │   │   ├── billing
│   │   │   │   │   │   │   │   ├── billing-section.module.css
│   │   │   │   │   │   │   │   ├── billing-section.tsx
│   │   │   │   │   │   │   │   ├── black-section.module.css
│   │   │   │   │   │   │   │   ├── black-section.tsx
│   │   │   │   │   │   │   │   ├── black-waitlist-section.module.css
│   │   │   │   │   │   │   │   ├── index.tsx
│   │   │   │   │   │   │   │   ├── lite-section.module.css
│   │   │   │   │   │   │   │   ├── lite-section.tsx
│   │   │   │   │   │   │   │   ├── monthly-limit-section.module.css
│   │   │   │   │   │   │   │   ├── monthly-limit-section.tsx
│   │   │   │   │   │   │   │   ├── payment-section.module.css
│   │   │   │   │   │   │   │   ├── payment-section.tsx
│   │   │   │   │   │   │   │   ├── reload-section.module.css
│   │   │   │   │   │   │   │   └── reload-section.tsx
│   │   │   │   │   │   │   ├── graph-section.module.css
│   │   │   │   │   │   │   ├── graph-section.tsx
│   │   │   │   │   │   │   ├── index.tsx
│   │   │   │   │   │   │   ├── keys
│   │   │   │   │   │   │   │   ├── index.tsx
│   │   │   │   │   │   │   │   ├── key-section.module.css
│   │   │   │   │   │   │   │   └── key-section.tsx
│   │   │   │   │   │   │   ├── members
│   │   │   │   │   │   │   │   ├── index.tsx
│   │   │   │   │   │   │   │   ├── member-section.module.css
│   │   │   │   │   │   │   │   ├── member-section.tsx
│   │   │   │   │   │   │   │   ├── role-dropdown.css
│   │   │   │   │   │   │   │   └── role-dropdown.tsx
│   │   │   │   │   │   │   ├── model-section.module.css
│   │   │   │   │   │   │   ├── model-section.tsx
│   │   │   │   │   │   │   ├── new-user-section.module.css
│   │   │   │   │   │   │   ├── new-user-section.tsx
│   │   │   │   │   │   │   ├── provider-section.module.css
│   │   │   │   │   │   │   ├── provider-section.tsx
│   │   │   │   │   │   │   ├── settings
│   │   │   │   │   │   │   │   ├── index.tsx
│   │   │   │   │   │   │   │   ├── settings-section.module.css
│   │   │   │   │   │   │   │   └── settings-section.tsx
│   │   │   │   │   │   │   ├── usage-section.module.css
│   │   │   │   │   │   │   └── usage-section.tsx
│   │   │   │   │   │   ├── [id].css
│   │   │   │   │   │   └── [id].tsx
│   │   │   │   │   ├── workspace.css
│   │   │   │   │   ├── workspace-picker.css
│   │   │   │   │   ├── workspace-picker.tsx
│   │   │   │   │   ├── workspace.tsx
│   │   │   │   │   └── zen
│   │   │   │   │       ├── index.css
│   │   │   │   │       ├── index.tsx
│   │   │   │   │       ├── lite
│   │   │   │   │       │   └── v1
│   │   │   │   │       │       ├── chat
│   │   │   │   │       │       │   └── completions.ts
│   │   │   │   │       │       └── messages.ts
│   │   │   │   │       ├── util
│   │   │   │   │       │   ├── dataDumper.ts
│   │   │   │   │       │   ├── error.ts
│   │   │   │   │       │   ├── handler.ts
│   │   │   │   │       │   ├── logger.ts
│   │   │   │   │       │   ├── provider
│   │   │   │   │       │   │   ├── anthropic.ts
│   │   │   │   │       │   │   ├── google.ts
│   │   │   │   │       │   │   ├── openai-compatible.ts
│   │   │   │   │       │   │   ├── openai.ts
│   │   │   │   │       │   │   └── provider.ts
│   │   │   │   │       │   ├── rateLimiter.ts
│   │   │   │   │       │   ├── stickyProviderTracker.ts
│   │   │   │   │       │   └── trialLimiter.ts
│   │   │   │   │       └── v1
│   │   │   │   │           ├── chat
│   │   │   │   │           │   └── completions.ts
│   │   │   │   │           ├── messages.ts
│   │   │   │   │           ├── models
│   │   │   │   │           │   └── [model].ts
│   │   │   │   │           ├── models.ts
│   │   │   │   │           └── responses.ts
│   │   │   │   └── style
│   │   │   │       ├── base.css
│   │   │   │       ├── component
│   │   │   │       │   └── button.css
│   │   │   │       ├── index.css
│   │   │   │       ├── reset.css
│   │   │   │       └── token
│   │   │   │           ├── color.css
│   │   │   │           ├── font.css
│   │   │   │           └── space.css
│   │   │   ├── sst-env.d.ts
│   │   │   ├── test
│   │   │   │   └── rateLimiter.test.ts
│   │   │   ├── tsconfig.json
│   │   │   └── vite.config.ts
│   │   ├── core
│   │   │   ├── drizzle.config.ts
│   │   │   ├── migrations
│   │   │   │   ├── 20250902065410_fluffy_raza
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250903035359_serious_whistler
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250911133331_violet_loners
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250911141957_dusty_clint_barton
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250911214917_first_mockingbird
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250911231144_jazzy_skrulls
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250912021148_parallel_gauntlet
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250912161749_familiar_nightshade
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250914213824_eminent_ultimatum
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250914222302_redundant_piledriver
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250914232505_needy_sue_storm
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250915150801_freezing_phil_sheldon
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250915172014_bright_photon
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250915172258_absurd_hobgoblin
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250919135159_demonic_princess_powerful
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250921042124_cloudy_revanche
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250923213126_cold_la_nuit
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250924230623_woozy_thaddeus_ross
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250928163425_nervous_iron_lad
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250928235456_dazzling_cable
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250929181457_supreme_jack_power
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20250929224703_flawless_clea
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251002175032_nice_dreadnoughts
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251002223020_optimal_paibok
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251003202205_early_black_crow
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251003210411_legal_joseph
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251004030300_numerous_prodigy
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251004045106_hot_wong
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251007024345_careful_cerise
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251007043715_panoramic_harrier
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251007230438_ordinary_ultragirl
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251008161718_outgoing_outlaw_kid
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251009021849_white_doctor_doom
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251016175624_cynical_jack_flag
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251016214520_short_bulldozer
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251017015733_narrow_blindfold
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251017024232_slimy_energizer
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251031163113_messy_jackal
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251125223403_famous_magik
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20251228182259_striped_forge
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260105034337_broken_gamora
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260106204919_odd_misty_knight
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260107000117_flat_nightmare
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260107022356_lame_calypso
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260107041522_tiny_captain_midlands
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260107055817_cuddly_diamondback
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260108224422_charming_black_bolt
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260109000245_huge_omega_red
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260109001625_mean_frank_castle
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260109014234_noisy_domino
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260109040130_bumpy_mephistopheles
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260113215232_jazzy_green_goblin
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260113223840_aromatic_agent_zero
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260116213606_gigantic_hardball
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260116224745_numerous_annihilus
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260122190905_moaning_karnak
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   ├── 20260222233442_clever_toxin
│   │   │   │   │   ├── migration.sql
│   │   │   │   │   └── snapshot.json
│   │   │   │   └── 20260224043338_nifty_starjammers
│   │   │   │       ├── migration.sql
│   │   │   │       └── snapshot.json
│   │   │   ├── package.json
│   │   │   ├── script
│   │   │   │   ├── black-cancel-waitlist.ts
│   │   │   │   ├── black-gift.ts
│   │   │   │   ├── black-onboard-waitlist.ts
│   │   │   │   ├── black-select-workspaces.ts
│   │   │   │   ├── black-transfer.ts
│   │   │   │   ├── credit-workspace.ts
│   │   │   │   ├── disable-reload.ts
│   │   │   │   ├── lookup-user.ts
│   │   │   │   ├── promote-black.ts
│   │   │   │   ├── promote-lite.ts
│   │   │   │   ├── promote-models.ts
│   │   │   │   ├── pull-models.ts
│   │   │   │   ├── reset-db.ts
│   │   │   │   ├── update-black.ts
│   │   │   │   ├── update-lite.ts
│   │   │   │   └── update-models.ts
│   │   │   ├── src
│   │   │   │   ├── account.ts
│   │   │   │   ├── actor.ts
│   │   │   │   ├── aws.ts
│   │   │   │   ├── billing.ts
│   │   │   │   ├── black.ts
│   │   │   │   ├── context.ts
│   │   │   │   ├── drizzle
│   │   │   │   │   ├── index.ts
│   │   │   │   │   └── types.ts
│   │   │   │   ├── identifier.ts
│   │   │   │   ├── key.ts
│   │   │   │   ├── lite.ts
│   │   │   │   ├── model.ts
│   │   │   │   ├── provider.ts
│   │   │   │   ├── schema
│   │   │   │   │   ├── account.sql.ts
│   │   │   │   │   ├── auth.sql.ts
│   │   │   │   │   ├── benchmark.sql.ts
│   │   │   │   │   ├── billing.sql.ts
│   │   │   │   │   ├── ip.sql.ts
│   │   │   │   │   ├── key.sql.ts
│   │   │   │   │   ├── model.sql.ts
│   │   │   │   │   ├── provider.sql.ts
│   │   │   │   │   ├── user.sql.ts
│   │   │   │   │   └── workspace.sql.ts
│   │   │   │   ├── subscription.ts
│   │   │   │   ├── user.ts
│   │   │   │   ├── util
│   │   │   │   │   ├── date.ts
│   │   │   │   │   ├── env.cloudflare.ts
│   │   │   │   │   ├── fn.ts
│   │   │   │   │   ├── log.ts
│   │   │   │   │   ├── memo.ts
│   │   │   │   │   └── price.ts
│   │   │   │   └── workspace.ts
│   │   │   ├── sst-env.d.ts
│   │   │   ├── test
│   │   │   │   ├── date.test.ts
│   │   │   │   └── subscription.test.ts
│   │   │   └── tsconfig.json
│   │   ├── function
│   │   │   ├── package.json
│   │   │   ├── src
│   │   │   │   ├── auth.ts
│   │   │   │   └── log-processor.ts
│   │   │   ├── sst-env.d.ts
│   │   │   └── tsconfig.json
│   │   ├── mail
│   │   │   ├── emails
│   │   │   │   ├── components.tsx
│   │   │   │   ├── styles.ts
│   │   │   │   └── templates
│   │   │   │       ├── InviteEmail.tsx
│   │   │   │       └── static
│   │   │   │           ├── ibm-plex-mono-latin-400.woff2
│   │   │   │           ├── ibm-plex-mono-latin-500.woff2
│   │   │   │           ├── ibm-plex-mono-latin-600.woff2
│   │   │   │           ├── ibm-plex-mono-latin-700.woff2
│   │   │   │           ├── JetBrainsMono-Medium.woff2
│   │   │   │           ├── JetBrainsMono-Regular.woff2
│   │   │   │           ├── logo.png
│   │   │   │           ├── right-arrow.png
│   │   │   │           ├── rubik-latin.woff2
│   │   │   │           └── zen-logo.png
│   │   │   ├── package.json
│   │   │   └── sst-env.d.ts
│   │   └── resource
│   │       ├── bun.lock
│   │       ├── package.json
│   │       ├── resource.cloudflare.ts
│   │       ├── resource.node.ts
│   │       ├── sst-env.d.ts
│   │       └── tsconfig.json
│   ├── containers
│   │   ├── base
│   │   │   └── Dockerfile
│   │   ├── bun-node
│   │   │   └── Dockerfile
│   │   ├── publish
│   │   │   └── Dockerfile
│   │   ├── README.md
│   │   ├── rust
│   │   │   └── Dockerfile
│   │   ├── script
│   │   │   └── build.ts
│   │   ├── tauri-linux
│   │   │   └── Dockerfile
│   │   └── tsconfig.json
│   ├── desktop
│   │   ├── AGENTS.md
│   │   ├── index.html
│   │   ├── package.json
│   │   ├── README.md
│   │   ├── scripts
│   │   │   ├── copy-bundles.ts
│   │   │   ├── predev.ts
│   │   │   ├── prepare.ts
│   │   │   └── utils.ts
│   │   ├── src
│   │   │   ├── bindings.ts
│   │   │   ├── cli.ts
│   │   │   ├── entry.tsx
│   │   │   ├── i18n
│   │   │   │   ├── ar.ts
│   │   │   │   ├── br.ts
│   │   │   │   ├── bs.ts
│   │   │   │   ├── da.ts
│   │   │   │   ├── de.ts
│   │   │   │   ├── en.ts
│   │   │   │   ├── es.ts
│   │   │   │   ├── fr.ts
│   │   │   │   ├── index.ts
│   │   │   │   ├── ja.ts
│   │   │   │   ├── ko.ts
│   │   │   │   ├── no.ts
│   │   │   │   ├── pl.ts
│   │   │   │   ├── ru.ts
│   │   │   │   ├── zh.ts
│   │   │   │   └── zht.ts
│   │   │   ├── index.tsx
│   │   │   ├── loading.tsx
│   │   │   ├── menu.ts
│   │   │   ├── styles.css
│   │   │   ├── updater.ts
│   │   │   └── webview-zoom.ts
│   │   ├── src-tauri
│   │   │   ├── assets
│   │   │   │   ├── nsis-header.bmp
│   │   │   │   └── nsis-sidebar.bmp
│   │   │   ├── build.rs
│   │   │   ├── capabilities
│   │   │   │   └── default.json
│   │   │   ├── Cargo.lock
│   │   │   ├── Cargo.toml
│   │   │   ├── entitlements.plist
│   │   │   ├── icons
│   │   │   │   ├── beta
│   │   │   │   │   ├── 128x128@2x.png
│   │   │   │   │   ├── 128x128.png
│   │   │   │   │   ├── 32x32.png
│   │   │   │   │   ├── 64x64.png
│   │   │   │   │   ├── android
│   │   │   │   │   │   ├── mipmap-anydpi-v26
│   │   │   │   │   │   │   └── ic_launcher.xml
│   │   │   │   │   │   ├── mipmap-hdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-mdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-xhdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-xxhdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-xxxhdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   └── values
│   │   │   │   │   │       └── ic_launcher_background.xml
│   │   │   │   │   ├── icon.icns
│   │   │   │   │   ├── icon.ico
│   │   │   │   │   ├── icon.png
│   │   │   │   │   ├── ios
│   │   │   │   │   │   ├── AppIcon-20x20@1x.png
│   │   │   │   │   │   ├── AppIcon-20x20@2x-1.png
│   │   │   │   │   │   ├── AppIcon-20x20@2x.png
│   │   │   │   │   │   ├── AppIcon-20x20@3x.png
│   │   │   │   │   │   ├── AppIcon-29x29@1x.png
│   │   │   │   │   │   ├── AppIcon-29x29@2x-1.png
│   │   │   │   │   │   ├── AppIcon-29x29@2x.png
│   │   │   │   │   │   ├── AppIcon-29x29@3x.png
│   │   │   │   │   │   ├── AppIcon-40x40@1x.png
│   │   │   │   │   │   ├── AppIcon-40x40@2x-1.png
│   │   │   │   │   │   ├── AppIcon-40x40@2x.png
│   │   │   │   │   │   ├── AppIcon-40x40@3x.png
│   │   │   │   │   │   ├── AppIcon-512@2x.png
│   │   │   │   │   │   ├── AppIcon-60x60@2x.png
│   │   │   │   │   │   ├── AppIcon-60x60@3x.png
│   │   │   │   │   │   ├── AppIcon-76x76@1x.png
│   │   │   │   │   │   ├── AppIcon-76x76@2x.png
│   │   │   │   │   │   └── AppIcon-83.5x83.5@2x.png
│   │   │   │   │   ├── Square107x107Logo.png
│   │   │   │   │   ├── Square142x142Logo.png
│   │   │   │   │   ├── Square150x150Logo.png
│   │   │   │   │   ├── Square284x284Logo.png
│   │   │   │   │   ├── Square30x30Logo.png
│   │   │   │   │   ├── Square310x310Logo.png
│   │   │   │   │   ├── Square44x44Logo.png
│   │   │   │   │   ├── Square71x71Logo.png
│   │   │   │   │   ├── Square89x89Logo.png
│   │   │   │   │   └── StoreLogo.png
│   │   │   │   ├── dev
│   │   │   │   │   ├── 128x128@2x.png
│   │   │   │   │   ├── 128x128.png
│   │   │   │   │   ├── 32x32.png
│   │   │   │   │   ├── 64x64.png
│   │   │   │   │   ├── android
│   │   │   │   │   │   ├── mipmap-anydpi-v26
│   │   │   │   │   │   │   └── ic_launcher.xml
│   │   │   │   │   │   ├── mipmap-hdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-mdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-xhdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-xxhdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-xxxhdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   └── values
│   │   │   │   │   │       └── ic_launcher_background.xml
│   │   │   │   │   ├── icon.icns
│   │   │   │   │   ├── icon.ico
│   │   │   │   │   ├── icon.png
│   │   │   │   │   ├── ios
│   │   │   │   │   │   ├── AppIcon-20x20@1x.png
│   │   │   │   │   │   ├── AppIcon-20x20@2x-1.png
│   │   │   │   │   │   ├── AppIcon-20x20@2x.png
│   │   │   │   │   │   ├── AppIcon-20x20@3x.png
│   │   │   │   │   │   ├── AppIcon-29x29@1x.png
│   │   │   │   │   │   ├── AppIcon-29x29@2x-1.png
│   │   │   │   │   │   ├── AppIcon-29x29@2x.png
│   │   │   │   │   │   ├── AppIcon-29x29@3x.png
│   │   │   │   │   │   ├── AppIcon-40x40@1x.png
│   │   │   │   │   │   ├── AppIcon-40x40@2x-1.png
│   │   │   │   │   │   ├── AppIcon-40x40@2x.png
│   │   │   │   │   │   ├── AppIcon-40x40@3x.png
│   │   │   │   │   │   ├── AppIcon-512@2x.png
│   │   │   │   │   │   ├── AppIcon-60x60@2x.png
│   │   │   │   │   │   ├── AppIcon-60x60@3x.png
│   │   │   │   │   │   ├── AppIcon-76x76@1x.png
│   │   │   │   │   │   ├── AppIcon-76x76@2x.png
│   │   │   │   │   │   └── AppIcon-83.5x83.5@2x.png
│   │   │   │   │   ├── Square107x107Logo.png
│   │   │   │   │   ├── Square142x142Logo.png
│   │   │   │   │   ├── Square150x150Logo.png
│   │   │   │   │   ├── Square284x284Logo.png
│   │   │   │   │   ├── Square30x30Logo.png
│   │   │   │   │   ├── Square310x310Logo.png
│   │   │   │   │   ├── Square44x44Logo.png
│   │   │   │   │   ├── Square71x71Logo.png
│   │   │   │   │   ├── Square89x89Logo.png
│   │   │   │   │   └── StoreLogo.png
│   │   │   │   ├── prod
│   │   │   │   │   ├── 128x128@2x.png
│   │   │   │   │   ├── 128x128.png
│   │   │   │   │   ├── 32x32.png
│   │   │   │   │   ├── 64x64.png
│   │   │   │   │   ├── android
│   │   │   │   │   │   ├── mipmap-anydpi-v26
│   │   │   │   │   │   │   └── ic_launcher.xml
│   │   │   │   │   │   ├── mipmap-hdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-mdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-xhdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-xxhdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   ├── mipmap-xxxhdpi
│   │   │   │   │   │   │   ├── ic_launcher_foreground.png
│   │   │   │   │   │   │   ├── ic_launcher.png
│   │   │   │   │   │   │   └── ic_launcher_round.png
│   │   │   │   │   │   └── values
│   │   │   │   │   │       └── ic_launcher_background.xml
│   │   │   │   │   ├── icon.icns
│   │   │   │   │   ├── icon.ico
│   │   │   │   │   ├── icon.png
│   │   │   │   │   ├── ios
│   │   │   │   │   │   ├── AppIcon-20x20@1x.png
│   │   │   │   │   │   ├── AppIcon-20x20@2x-1.png
│   │   │   │   │   │   ├── AppIcon-20x20@2x.png
│   │   │   │   │   │   ├── AppIcon-20x20@3x.png
│   │   │   │   │   │   ├── AppIcon-29x29@1x.png
│   │   │   │   │   │   ├── AppIcon-29x29@2x-1.png
│   │   │   │   │   │   ├── AppIcon-29x29@2x.png
│   │   │   │   │   │   ├── AppIcon-29x29@3x.png
│   │   │   │   │   │   ├── AppIcon-40x40@1x.png
│   │   │   │   │   │   ├── AppIcon-40x40@2x-1.png
│   │   │   │   │   │   ├── AppIcon-40x40@2x.png
│   │   │   │   │   │   ├── AppIcon-40x40@3x.png
│   │   │   │   │   │   ├── AppIcon-512@2x.png
│   │   │   │   │   │   ├── AppIcon-60x60@2x.png
│   │   │   │   │   │   ├── AppIcon-60x60@3x.png
│   │   │   │   │   │   ├── AppIcon-76x76@1x.png
│   │   │   │   │   │   ├── AppIcon-76x76@2x.png
│   │   │   │   │   │   └── AppIcon-83.5x83.5@2x.png
│   │   │   │   │   ├── Square107x107Logo.png
│   │   │   │   │   ├── Square142x142Logo.png
│   │   │   │   │   ├── Square150x150Logo.png
│   │   │   │   │   ├── Square284x284Logo.png
│   │   │   │   │   ├── Square30x30Logo.png
│   │   │   │   │   ├── Square310x310Logo.png
│   │   │   │   │   ├── Square44x44Logo.png
│   │   │   │   │   ├── Square71x71Logo.png
│   │   │   │   │   ├── Square89x89Logo.png
│   │   │   │   │   └── StoreLogo.png
│   │   │   │   └── README.md
│   │   │   ├── release
│   │   │   │   └── appstream.metainfo.xml
│   │   │   ├── src
│   │   │   │   ├── cli.rs
│   │   │   │   ├── constants.rs
│   │   │   │   ├── lib.rs
│   │   │   │   ├── linux_display.rs
│   │   │   │   ├── linux_windowing.rs
│   │   │   │   ├── logging.rs
│   │   │   │   ├── main.rs
│   │   │   │   ├── markdown.rs
│   │   │   │   ├── server.rs
│   │   │   │   ├── window_customizer.rs
│   │   │   │   └── windows.rs
│   │   │   ├── tauri.beta.conf.json
│   │   │   ├── tauri.conf.json
│   │   │   └── tauri.prod.conf.json
│   │   ├── sst-env.d.ts
│   │   ├── tsconfig.json
│   │   └── vite.config.ts
│   ├── docs
│   │   ├── ai-tools
│   │   │   ├── claude-code.mdx
│   │   │   ├── cursor.mdx
│   │   │   └── windsurf.mdx
│   │   ├── development.mdx
│   │   ├── docs.json
│   │   ├── essentials
│   │   │   ├── code.mdx
│   │   │   ├── images.mdx
│   │   │   ├── markdown.mdx
│   │   │   ├── navigation.mdx
│   │   │   ├── reusable-snippets.mdx
│   │   │   └── settings.mdx
│   │   ├── favicon.svg
│   │   ├── favicon-v3.svg
│   │   ├── images
│   │   │   ├── checks-passed.png
│   │   │   ├── hero-dark.png
│   │   │   └── hero-light.png
│   │   ├── index.mdx
│   │   ├── LICENSE
│   │   ├── logo
│   │   │   ├── dark.svg
│   │   │   └── light.svg
│   │   ├── openapi.json -> ../sdk/openapi.json
│   │   ├── quickstart.mdx
│   │   ├── README.md
│   │   └── snippets
│   │       └── snippet-intro.mdx
│   ├── enterprise
│   │   ├── package.json
│   │   ├── public
│   │   │   ├── apple-touch-icon.png -> ../../ui/src/assets/favicon/apple-touch-icon.png
│   │   │   ├── apple-touch-icon-v3.png -> ../../ui/src/assets/favicon/apple-touch-icon-v3.png
│   │   │   ├── favicon-96x96.png -> ../../ui/src/assets/favicon/favicon-96x96.png
│   │   │   ├── favicon-96x96-v3.png -> ../../ui/src/assets/favicon/favicon-96x96-v3.png
│   │   │   ├── favicon.ico -> ../../ui/src/assets/favicon/favicon.ico
│   │   │   ├── favicon.svg -> ../../ui/src/assets/favicon/favicon.svg
│   │   │   ├── favicon-v3.ico -> ../../ui/src/assets/favicon/favicon-v3.ico
│   │   │   ├── favicon-v3.svg -> ../../ui/src/assets/favicon/favicon-v3.svg
│   │   │   ├── site.webmanifest -> ../../ui/src/assets/favicon/site.webmanifest
│   │   │   ├── social-share.png -> ../../ui/src/assets/images/social-share.png
│   │   │   ├── social-share-zen.png -> ../../ui/src/assets/images/social-share-zen.png
│   │   │   ├── web-app-manifest-192x192.png -> ../../ui/src/assets/favicon/web-app-manifest-192x192.png
│   │   │   └── web-app-manifest-512x512.png -> ../../ui/src/assets/favicon/web-app-manifest-512x512.png
│   │   ├── README.md
│   │   ├── script
│   │   │   └── scrap.ts
│   │   ├── src
│   │   │   ├── app.css
│   │   │   ├── app.tsx
│   │   │   ├── core
│   │   │   │   ├── share.ts
│   │   │   │   └── storage.ts
│   │   │   ├── custom-elements.d.ts -> ../../ui/src/custom-elements.d.ts
│   │   │   ├── entry-client.tsx
│   │   │   ├── entry-server.tsx
│   │   │   ├── global.d.ts
│   │   │   └── routes
│   │   │       ├── [...404].tsx
│   │   │       ├── api
│   │   │       │   └── [...path].ts
│   │   │       ├── index.tsx
│   │   │       ├── share
│   │   │       │   └── [shareID].tsx
│   │   │       └── share.tsx
│   │   ├── sst-env.d.ts
│   │   ├── test
│   │   │   └── core
│   │   │       ├── share.test.ts
│   │   │       └── storage.test.ts
│   │   ├── test-debug.ts
│   │   ├── tsconfig.json
│   │   └── vite.config.ts
│   ├── extensions
│   │   └── zed
│   │       ├── extension.toml
│   │       ├── icons
│   │       │   └── opencode.svg
│   │       └── LICENSE -> ../../../LICENSE
│   ├── function
│   │   ├── package.json
│   │   ├── src
│   │   │   └── api.ts
│   │   ├── sst-env.d.ts
│   │   └── tsconfig.json
│   ├── identity
│   │   ├── mark-192x192.png
│   │   ├── mark-512x512-light.png
│   │   ├── mark-512x512.png
│   │   ├── mark-96x96.png
│   │   ├── mark-light.svg
│   │   └── mark.svg
│   ├── opencode
│   │   ├── AGENTS.md
│   │   ├── bin
│   │   │   └── opencode
│   │   ├── bunfig.toml
│   │   ├── Dockerfile
│   │   ├── drizzle.config.ts
│   │   ├── migration
│   │   │   ├── 20260127222353_familiar_lady_ursula
│   │   │   │   ├── migration.sql
│   │   │   │   └── snapshot.json
│   │   │   ├── 20260211171708_add_project_commands
│   │   │   │   ├── migration.sql
│   │   │   │   └── snapshot.json
│   │   │   └── 20260213144116_wakeful_the_professor
│   │   │       ├── migration.sql
│   │   │       └── snapshot.json
│   │   ├── package.json
│   │   ├── parsers-config.ts
│   │   ├── README.md
│   │   ├── script
│   │   │   ├── build.ts
│   │   │   ├── check-migrations.ts
│   │   │   ├── postinstall.mjs
│   │   │   ├── publish.ts
│   │   │   ├── schema.ts
│   │   │   └── seed-e2e.ts
│   │   ├── src
│   │   │   ├── acp
│   │   │   │   ├── agent.ts
│   │   │   │   ├── README.md
│   │   │   │   ├── session.ts
│   │   │   │   └── types.ts
│   │   │   ├── agent
│   │   │   │   ├── agent.ts
│   │   │   │   ├── generate.txt
│   │   │   │   └── prompt
│   │   │   │       ├── compaction.txt
│   │   │   │       ├── explore.txt
│   │   │   │       ├── summary.txt
│   │   │   │       └── title.txt
│   │   │   ├── auth
│   │   │   │   └── index.ts
│   │   │   ├── bun
│   │   │   │   ├── index.ts
│   │   │   │   └── registry.ts
│   │   │   ├── bus
│   │   │   │   ├── bus-event.ts
│   │   │   │   ├── global.ts
│   │   │   │   └── index.ts
│   │   │   ├── cli
│   │   │   │   ├── bootstrap.ts
│   │   │   │   ├── cmd
│   │   │   │   │   ├── acp.ts
│   │   │   │   │   ├── agent.ts
│   │   │   │   │   ├── auth.ts
│   │   │   │   │   ├── cmd.ts
│   │   │   │   │   ├── db.ts
│   │   │   │   │   ├── debug
│   │   │   │   │   │   ├── agent.ts
│   │   │   │   │   │   ├── config.ts
│   │   │   │   │   │   ├── file.ts
│   │   │   │   │   │   ├── index.ts
│   │   │   │   │   │   ├── lsp.ts
│   │   │   │   │   │   ├── ripgrep.ts
│   │   │   │   │   │   ├── scrap.ts
│   │   │   │   │   │   ├── skill.ts
│   │   │   │   │   │   └── snapshot.ts
│   │   │   │   │   ├── export.ts
│   │   │   │   │   ├── generate.ts
│   │   │   │   │   ├── github.ts
│   │   │   │   │   ├── import.ts
│   │   │   │   │   ├── mcp.ts
│   │   │   │   │   ├── models.ts
│   │   │   │   │   ├── pr.ts
│   │   │   │   │   ├── run.ts
│   │   │   │   │   ├── serve.ts
│   │   │   │   │   ├── session.ts
│   │   │   │   │   ├── stats.ts
│   │   │   │   │   ├── tui
│   │   │   │   │   │   ├── app.tsx
│   │   │   │   │   │   ├── attach.ts
│   │   │   │   │   │   ├── component
│   │   │   │   │   │   │   ├── border.tsx
│   │   │   │   │   │   │   ├── dialog-agent.tsx
│   │   │   │   │   │   │   ├── dialog-command.tsx
│   │   │   │   │   │   │   ├── dialog-mcp.tsx
│   │   │   │   │   │   │   ├── dialog-model.tsx
│   │   │   │   │   │   │   ├── dialog-provider.tsx
│   │   │   │   │   │   │   ├── dialog-session-list.tsx
│   │   │   │   │   │   │   ├── dialog-session-rename.tsx
│   │   │   │   │   │   │   ├── dialog-skill.tsx
│   │   │   │   │   │   │   ├── dialog-stash.tsx
│   │   │   │   │   │   │   ├── dialog-status.tsx
│   │   │   │   │   │   │   ├── dialog-tag.tsx
│   │   │   │   │   │   │   ├── dialog-theme-list.tsx
│   │   │   │   │   │   │   ├── logo.tsx
│   │   │   │   │   │   │   ├── prompt
│   │   │   │   │   │   │   │   ├── autocomplete.tsx
│   │   │   │   │   │   │   │   ├── frecency.tsx
│   │   │   │   │   │   │   │   ├── history.tsx
│   │   │   │   │   │   │   │   ├── index.tsx
│   │   │   │   │   │   │   │   └── stash.tsx
│   │   │   │   │   │   │   ├── spinner.tsx
│   │   │   │   │   │   │   ├── textarea-keybindings.ts
│   │   │   │   │   │   │   ├── tips.tsx
│   │   │   │   │   │   │   └── todo-item.tsx
│   │   │   │   │   │   ├── context
│   │   │   │   │   │   │   ├── args.tsx
│   │   │   │   │   │   │   ├── directory.ts
│   │   │   │   │   │   │   ├── exit.tsx
│   │   │   │   │   │   │   ├── helper.tsx
│   │   │   │   │   │   │   ├── keybind.tsx
│   │   │   │   │   │   │   ├── kv.tsx
│   │   │   │   │   │   │   ├── local.tsx
│   │   │   │   │   │   │   ├── prompt.tsx
│   │   │   │   │   │   │   ├── route.tsx
│   │   │   │   │   │   │   ├── sdk.tsx
│   │   │   │   │   │   │   ├── sync.tsx
│   │   │   │   │   │   │   ├── theme
│   │   │   │   │   │   │   │   ├── aura.json
│   │   │   │   │   │   │   │   ├── ayu.json
│   │   │   │   │   │   │   │   ├── carbonfox.json
│   │   │   │   │   │   │   │   ├── catppuccin-frappe.json
│   │   │   │   │   │   │   │   ├── catppuccin.json
│   │   │   │   │   │   │   │   ├── catppuccin-macchiato.json
│   │   │   │   │   │   │   │   ├── cobalt2.json
│   │   │   │   │   │   │   │   ├── cursor.json
│   │   │   │   │   │   │   │   ├── dracula.json
│   │   │   │   │   │   │   │   ├── everforest.json
│   │   │   │   │   │   │   │   ├── flexoki.json
│   │   │   │   │   │   │   │   ├── github.json
│   │   │   │   │   │   │   │   ├── gruvbox.json
│   │   │   │   │   │   │   │   ├── kanagawa.json
│   │   │   │   │   │   │   │   ├── lucent-orng.json
│   │   │   │   │   │   │   │   ├── material.json
│   │   │   │   │   │   │   │   ├── matrix.json
│   │   │   │   │   │   │   │   ├── mercury.json
│   │   │   │   │   │   │   │   ├── monokai.json
│   │   │   │   │   │   │   │   ├── nightowl.json
│   │   │   │   │   │   │   │   ├── nord.json
│   │   │   │   │   │   │   │   ├── one-dark.json
│   │   │   │   │   │   │   │   ├── opencode.json
│   │   │   │   │   │   │   │   ├── orng.json
│   │   │   │   │   │   │   │   ├── osaka-jade.json
│   │   │   │   │   │   │   │   ├── palenight.json
│   │   │   │   │   │   │   │   ├── rosepine.json
│   │   │   │   │   │   │   │   ├── solarized.json
│   │   │   │   │   │   │   │   ├── synthwave84.json
│   │   │   │   │   │   │   │   ├── tokyonight.json
│   │   │   │   │   │   │   │   ├── vercel.json
│   │   │   │   │   │   │   │   ├── vesper.json
│   │   │   │   │   │   │   │   └── zenburn.json
│   │   │   │   │   │   │   └── theme.tsx
│   │   │   │   │   │   ├── event.ts
│   │   │   │   │   │   ├── routes
│   │   │   │   │   │   │   ├── home.tsx
│   │   │   │   │   │   │   └── session
│   │   │   │   │   │   │       ├── dialog-fork-from-timeline.tsx
│   │   │   │   │   │   │       ├── dialog-message.tsx
│   │   │   │   │   │   │       ├── dialog-subagent.tsx
│   │   │   │   │   │   │       ├── dialog-timeline.tsx
│   │   │   │   │   │   │       ├── footer.tsx
│   │   │   │   │   │   │       ├── header.tsx
│   │   │   │   │   │   │       ├── index.tsx
│   │   │   │   │   │   │       ├── permission.tsx
│   │   │   │   │   │   │       ├── question.tsx
│   │   │   │   │   │   │       └── sidebar.tsx
│   │   │   │   │   │   ├── thread.ts
│   │   │   │   │   │   ├── ui
│   │   │   │   │   │   │   ├── dialog-alert.tsx
│   │   │   │   │   │   │   ├── dialog-confirm.tsx
│   │   │   │   │   │   │   ├── dialog-export-options.tsx
│   │   │   │   │   │   │   ├── dialog-help.tsx
│   │   │   │   │   │   │   ├── dialog-prompt.tsx
│   │   │   │   │   │   │   ├── dialog-select.tsx
│   │   │   │   │   │   │   ├── dialog.tsx
│   │   │   │   │   │   │   ├── link.tsx
│   │   │   │   │   │   │   ├── spinner.ts
│   │   │   │   │   │   │   └── toast.tsx
│   │   │   │   │   │   ├── util
│   │   │   │   │   │   │   ├── clipboard.ts
│   │   │   │   │   │   │   ├── editor.ts
│   │   │   │   │   │   │   ├── selection.ts
│   │   │   │   │   │   │   ├── signal.ts
│   │   │   │   │   │   │   ├── terminal.ts
│   │   │   │   │   │   │   └── transcript.ts
│   │   │   │   │   │   ├── win32.ts
│   │   │   │   │   │   └── worker.ts
│   │   │   │   │   ├── uninstall.ts
│   │   │   │   │   ├── upgrade.ts
│   │   │   │   │   ├── web.ts
│   │   │   │   │   └── workspace-serve.ts
│   │   │   │   ├── error.ts
│   │   │   │   ├── logo.ts
│   │   │   │   ├── network.ts
│   │   │   │   ├── ui.ts
│   │   │   │   └── upgrade.ts
│   │   │   ├── command
│   │   │   │   ├── index.ts
│   │   │   │   └── template
│   │   │   │       ├── initialize.txt
│   │   │   │       └── review.txt
│   │   │   ├── config
│   │   │   │   ├── config.ts
│   │   │   │   └── markdown.ts
│   │   │   ├── control
│   │   │   │   ├── control.sql.ts
│   │   │   │   └── index.ts
│   │   │   ├── env
│   │   │   │   └── index.ts
│   │   │   ├── file
│   │   │   │   ├── ignore.ts
│   │   │   │   ├── index.ts
│   │   │   │   ├── ripgrep.ts
│   │   │   │   ├── time.ts
│   │   │   │   └── watcher.ts
│   │   │   ├── flag
│   │   │   │   └── flag.ts
│   │   │   ├── format
│   │   │   │   ├── formatter.ts
│   │   │   │   └── index.ts
│   │   │   ├── global
│   │   │   │   └── index.ts
│   │   │   ├── id
│   │   │   │   └── id.ts
│   │   │   ├── ide
│   │   │   │   └── index.ts
│   │   │   ├── index.ts
│   │   │   ├── installation
│   │   │   │   └── index.ts
│   │   │   ├── lsp
│   │   │   │   ├── client.ts
│   │   │   │   ├── index.ts
│   │   │   │   ├── language.ts
│   │   │   │   └── server.ts
│   │   │   ├── mcp
│   │   │   │   ├── auth.ts
│   │   │   │   ├── index.ts
│   │   │   │   ├── oauth-callback.ts
│   │   │   │   └── oauth-provider.ts
│   │   │   ├── patch
│   │   │   │   └── index.ts
│   │   │   ├── permission
│   │   │   │   ├── arity.ts
│   │   │   │   ├── index.ts
│   │   │   │   └── next.ts
│   │   │   ├── plugin
│   │   │   │   ├── codex.ts
│   │   │   │   ├── copilot.ts
│   │   │   │   └── index.ts
│   │   │   ├── project
│   │   │   │   ├── bootstrap.ts
│   │   │   │   ├── instance.ts
│   │   │   │   ├── project.sql.ts
│   │   │   │   ├── project.ts
│   │   │   │   ├── state.ts
│   │   │   │   └── vcs.ts
│   │   │   ├── provider
│   │   │   │   ├── auth.ts
│   │   │   │   ├── error.ts
│   │   │   │   ├── models.ts
│   │   │   │   ├── provider.ts
│   │   │   │   ├── sdk
│   │   │   │   │   └── copilot
│   │   │   │   │       ├── chat
│   │   │   │   │       │   ├── convert-to-openai-compatible-chat-messages.ts
│   │   │   │   │       │   ├── get-response-metadata.ts
│   │   │   │   │       │   ├── map-openai-compatible-finish-reason.ts
│   │   │   │   │       │   ├── openai-compatible-api-types.ts
│   │   │   │   │       │   ├── openai-compatible-chat-language-model.ts
│   │   │   │   │       │   ├── openai-compatible-chat-options.ts
│   │   │   │   │       │   ├── openai-compatible-metadata-extractor.ts
│   │   │   │   │       │   └── openai-compatible-prepare-tools.ts
│   │   │   │   │       ├── copilot-provider.ts
│   │   │   │   │       ├── index.ts
│   │   │   │   │       ├── openai-compatible-error.ts
│   │   │   │   │       ├── README.md
│   │   │   │   │       └── responses
│   │   │   │   │           ├── convert-to-openai-responses-input.ts
│   │   │   │   │           ├── map-openai-responses-finish-reason.ts
│   │   │   │   │           ├── openai-config.ts
│   │   │   │   │           ├── openai-error.ts
│   │   │   │   │           ├── openai-responses-api-types.ts
│   │   │   │   │           ├── openai-responses-language-model.ts
│   │   │   │   │           ├── openai-responses-prepare-tools.ts
│   │   │   │   │           ├── openai-responses-settings.ts
│   │   │   │   │           └── tool
│   │   │   │   │               ├── code-interpreter.ts
│   │   │   │   │               ├── file-search.ts
│   │   │   │   │               ├── image-generation.ts
│   │   │   │   │               ├── local-shell.ts
│   │   │   │   │               ├── web-search-preview.ts
│   │   │   │   │               └── web-search.ts
│   │   │   │   └── transform.ts
│   │   │   ├── pty
│   │   │   │   └── index.ts
│   │   │   ├── question
│   │   │   │   └── index.ts
│   │   │   ├── scheduler
│   │   │   │   └── index.ts
│   │   │   ├── server
│   │   │   │   ├── error.ts
│   │   │   │   ├── event.ts
│   │   │   │   ├── mdns.ts
│   │   │   │   ├── routes
│   │   │   │   │   ├── config.ts
│   │   │   │   │   ├── experimental.ts
│   │   │   │   │   ├── file.ts
│   │   │   │   │   ├── global.ts
│   │   │   │   │   ├── mcp.ts
│   │   │   │   │   ├── permission.ts
│   │   │   │   │   ├── project.ts
│   │   │   │   │   ├── provider.ts
│   │   │   │   │   ├── pty.ts
│   │   │   │   │   ├── question.ts
│   │   │   │   │   ├── session.ts
│   │   │   │   │   └── tui.ts
│   │   │   │   └── server.ts
│   │   │   ├── session
│   │   │   │   ├── compaction.ts
│   │   │   │   ├── index.ts
│   │   │   │   ├── instruction.ts
│   │   │   │   ├── llm.ts
│   │   │   │   ├── message.ts
│   │   │   │   ├── message-v2.ts
│   │   │   │   ├── processor.ts
│   │   │   │   ├── prompt
│   │   │   │   │   ├── anthropic-20250930.txt
│   │   │   │   │   ├── anthropic.txt
│   │   │   │   │   ├── beast.txt
│   │   │   │   │   ├── build-switch.txt
│   │   │   │   │   ├── codex_header.txt
│   │   │   │   │   ├── copilot-gpt-5.txt
│   │   │   │   │   ├── gemini.txt
│   │   │   │   │   ├── max-steps.txt
│   │   │   │   │   ├── plan-reminder-anthropic.txt
│   │   │   │   │   ├── plan.txt
│   │   │   │   │   ├── qwen.txt
│   │   │   │   │   └── trinity.txt
│   │   │   │   ├── prompt.ts
│   │   │   │   ├── retry.ts
│   │   │   │   ├── revert.ts
│   │   │   │   ├── session.sql.ts
│   │   │   │   ├── status.ts
│   │   │   │   ├── summary.ts
│   │   │   │   ├── system.ts
│   │   │   │   └── todo.ts
│   │   │   ├── share
│   │   │   │   ├── share-next.ts
│   │   │   │   └── share.sql.ts
│   │   │   ├── shell
│   │   │   │   └── shell.ts
│   │   │   ├── skill
│   │   │   │   ├── discovery.ts
│   │   │   │   ├── index.ts
│   │   │   │   └── skill.ts
│   │   │   ├── snapshot
│   │   │   │   └── index.ts
│   │   │   ├── sql.d.ts
│   │   │   ├── storage
│   │   │   │   ├── db.ts
│   │   │   │   ├── json-migration.ts
│   │   │   │   ├── schema.sql.ts
│   │   │   │   ├── schema.ts
│   │   │   │   └── storage.ts
│   │   │   ├── tool
│   │   │   │   ├── apply_patch.ts
│   │   │   │   ├── apply_patch.txt
│   │   │   │   ├── bash.ts
│   │   │   │   ├── bash.txt
│   │   │   │   ├── batch.ts
│   │   │   │   ├── batch.txt
│   │   │   │   ├── codesearch.ts
│   │   │   │   ├── codesearch.txt
│   │   │   │   ├── edit.ts
│   │   │   │   ├── edit.txt
│   │   │   │   ├── external-directory.ts
│   │   │   │   ├── glob.ts
│   │   │   │   ├── glob.txt
│   │   │   │   ├── grep.ts
│   │   │   │   ├── grep.txt
│   │   │   │   ├── invalid.ts
│   │   │   │   ├── lsp.ts
│   │   │   │   ├── lsp.txt
│   │   │   │   ├── ls.ts
│   │   │   │   ├── ls.txt
│   │   │   │   ├── multiedit.ts
│   │   │   │   ├── multiedit.txt
│   │   │   │   ├── plan-enter.txt
│   │   │   │   ├── plan-exit.txt
│   │   │   │   ├── plan.ts
│   │   │   │   ├── question.ts
│   │   │   │   ├── question.txt
│   │   │   │   ├── read.ts
│   │   │   │   ├── read.txt
│   │   │   │   ├── registry.ts
│   │   │   │   ├── skill.ts
│   │   │   │   ├── task.ts
│   │   │   │   ├── task.txt
│   │   │   │   ├── todoread.txt
│   │   │   │   ├── todo.ts
│   │   │   │   ├── todowrite.txt
│   │   │   │   ├── tool.ts
│   │   │   │   ├── truncation.ts
│   │   │   │   ├── webfetch.ts
│   │   │   │   ├── webfetch.txt
│   │   │   │   ├── websearch.ts
│   │   │   │   ├── websearch.txt
│   │   │   │   ├── write.ts
│   │   │   │   └── write.txt
│   │   │   ├── util
│   │   │   │   ├── abort.ts
│   │   │   │   ├── archive.ts
│   │   │   │   ├── color.ts
│   │   │   │   ├── context.ts
│   │   │   │   ├── defer.ts
│   │   │   │   ├── eventloop.ts
│   │   │   │   ├── filesystem.ts
│   │   │   │   ├── fn.ts
│   │   │   │   ├── format.ts
│   │   │   │   ├── git.ts
│   │   │   │   ├── glob.ts
│   │   │   │   ├── iife.ts
│   │   │   │   ├── keybind.ts
│   │   │   │   ├── lazy.ts
│   │   │   │   ├── locale.ts
│   │   │   │   ├── lock.ts
│   │   │   │   ├── log.ts
│   │   │   │   ├── proxied.ts
│   │   │   │   ├── queue.ts
│   │   │   │   ├── rpc.ts
│   │   │   │   ├── scrap.ts
│   │   │   │   ├── signal.ts
│   │   │   │   ├── timeout.ts
│   │   │   │   ├── token.ts
│   │   │   │   └── wildcard.ts
│   │   │   └── worktree
│   │   │       └── index.ts
│   │   ├── sst-env.d.ts
│   │   ├── test
│   │   │   ├── acp
│   │   │   │   ├── agent-interface.test.ts
│   │   │   │   └── event-subscription.test.ts
│   │   │   ├── agent
│   │   │   │   └── agent.test.ts
│   │   │   ├── AGENTS.md
│   │   │   ├── bun.test.ts
│   │   │   ├── cli
│   │   │   │   ├── github-action.test.ts
│   │   │   │   ├── github-remote.test.ts
│   │   │   │   ├── import.test.ts
│   │   │   │   ├── plugin-auth-picker.test.ts
│   │   │   │   └── tui
│   │   │   │       └── transcript.test.ts
│   │   │   ├── config
│   │   │   │   ├── agent-color.test.ts
│   │   │   │   ├── config.test.ts
│   │   │   │   ├── fixtures
│   │   │   │   │   ├── empty-frontmatter.md
│   │   │   │   │   ├── frontmatter.md
│   │   │   │   │   ├── markdown-header.md
│   │   │   │   │   ├── no-frontmatter.md
│   │   │   │   │   └── weird-model-id.md
│   │   │   │   └── markdown.test.ts
│   │   │   ├── file
│   │   │   │   ├── ignore.test.ts
│   │   │   │   ├── index.test.ts
│   │   │   │   ├── path-traversal.test.ts
│   │   │   │   ├── ripgrep.test.ts
│   │   │   │   └── time.test.ts
│   │   │   ├── fixture
│   │   │   │   ├── fixture.ts
│   │   │   │   ├── lsp
│   │   │   │   │   └── fake-lsp-server.js
│   │   │   │   └── skills
│   │   │   │       ├── agents-sdk
│   │   │   │       │   ├── references
│   │   │   │       │   │   └── callable.md
│   │   │   │       │   └── SKILL.md
│   │   │   │       ├── cloudflare
│   │   │   │       │   └── SKILL.md
│   │   │   │       └── index.json
│   │   │   ├── ide
│   │   │   │   └── ide.test.ts
│   │   │   ├── keybind.test.ts
│   │   │   ├── lsp
│   │   │   │   └── client.test.ts
│   │   │   ├── mcp
│   │   │   │   ├── headers.test.ts
│   │   │   │   └── oauth-browser.test.ts
│   │   │   ├── memory
│   │   │   │   └── abort-leak.test.ts
│   │   │   ├── patch
│   │   │   │   └── patch.test.ts
│   │   │   ├── permission
│   │   │   │   ├── arity.test.ts
│   │   │   │   └── next.test.ts
│   │   │   ├── permission-task.test.ts
│   │   │   ├── plugin
│   │   │   │   ├── auth-override.test.ts
│   │   │   │   └── codex.test.ts
│   │   │   ├── preload.ts
│   │   │   ├── project
│   │   │   │   ├── project.test.ts
│   │   │   │   └── worktree-remove.test.ts
│   │   │   ├── provider
│   │   │   │   ├── amazon-bedrock.test.ts
│   │   │   │   ├── copilot
│   │   │   │   │   ├── convert-to-copilot-messages.test.ts
│   │   │   │   │   └── copilot-chat-model.test.ts
│   │   │   │   ├── gitlab-duo.test.ts
│   │   │   │   ├── provider.test.ts
│   │   │   │   └── transform.test.ts
│   │   │   ├── pty
│   │   │   │   └── pty-output-isolation.test.ts
│   │   │   ├── question
│   │   │   │   └── question.test.ts
│   │   │   ├── scheduler.test.ts
│   │   │   ├── server
│   │   │   │   ├── global-session-list.test.ts
│   │   │   │   ├── session-list.test.ts
│   │   │   │   └── session-select.test.ts
│   │   │   ├── session
│   │   │   │   ├── compaction.test.ts
│   │   │   │   ├── instruction.test.ts
│   │   │   │   ├── llm.test.ts
│   │   │   │   ├── message-v2.test.ts
│   │   │   │   ├── prompt.test.ts
│   │   │   │   ├── retry.test.ts
│   │   │   │   ├── revert-compact.test.ts
│   │   │   │   ├── session.test.ts
│   │   │   │   ├── structured-output-integration.test.ts
│   │   │   │   └── structured-output.test.ts
│   │   │   ├── skill
│   │   │   │   ├── discovery.test.ts
│   │   │   │   └── skill.test.ts
│   │   │   ├── snapshot
│   │   │   │   └── snapshot.test.ts
│   │   │   ├── storage
│   │   │   │   └── json-migration.test.ts
│   │   │   ├── tool
│   │   │   │   ├── apply_patch.test.ts
│   │   │   │   ├── bash.test.ts
│   │   │   │   ├── edit.test.ts
│   │   │   │   ├── external-directory.test.ts
│   │   │   │   ├── fixtures
│   │   │   │   │   ├── large-image.png
│   │   │   │   │   └── models-api.json
│   │   │   │   ├── grep.test.ts
│   │   │   │   ├── question.test.ts
│   │   │   │   ├── read.test.ts
│   │   │   │   ├── registry.test.ts
│   │   │   │   ├── skill.test.ts
│   │   │   │   ├── __snapshots__
│   │   │   │   │   └── tool.test.ts.snap
│   │   │   │   ├── truncation.test.ts
│   │   │   │   ├── webfetch.test.ts
│   │   │   │   └── write.test.ts
│   │   │   └── util
│   │   │       ├── filesystem.test.ts
│   │   │       ├── format.test.ts
│   │   │       ├── glob.test.ts
│   │   │       ├── iife.test.ts
│   │   │       ├── lazy.test.ts
│   │   │       ├── lock.test.ts
│   │   │       ├── timeout.test.ts
│   │   │       └── wildcard.test.ts
│   │   └── tsconfig.json
│   ├── plugin
│   │   ├── package.json
│   │   ├── script
│   │   │   └── publish.ts
│   │   ├── src
│   │   │   ├── example.ts
│   │   │   ├── index.ts
│   │   │   ├── shell.ts
│   │   │   └── tool.ts
│   │   ├── sst-env.d.ts
│   │   └── tsconfig.json
│   ├── script
│   │   ├── package.json
│   │   ├── src
│   │   │   └── index.ts
│   │   ├── sst-env.d.ts
│   │   └── tsconfig.json
│   ├── sdk
│   │   ├── js
│   │   │   ├── example
│   │   │   │   └── example.ts
│   │   │   ├── package.json
│   │   │   ├── script
│   │   │   │   ├── build.ts
│   │   │   │   └── publish.ts
│   │   │   ├── src
│   │   │   │   ├── client.ts
│   │   │   │   ├── gen
│   │   │   │   │   ├── client
│   │   │   │   │   │   ├── client.gen.ts
│   │   │   │   │   │   ├── index.ts
│   │   │   │   │   │   ├── types.gen.ts
│   │   │   │   │   │   └── utils.gen.ts
│   │   │   │   │   ├── client.gen.ts
│   │   │   │   │   ├── core
│   │   │   │   │   │   ├── auth.gen.ts
│   │   │   │   │   │   ├── bodySerializer.gen.ts
│   │   │   │   │   │   ├── params.gen.ts
│   │   │   │   │   │   ├── pathSerializer.gen.ts
│   │   │   │   │   │   ├── queryKeySerializer.gen.ts
│   │   │   │   │   │   ├── serverSentEvents.gen.ts
│   │   │   │   │   │   ├── types.gen.ts
│   │   │   │   │   │   └── utils.gen.ts
│   │   │   │   │   ├── sdk.gen.ts
│   │   │   │   │   └── types.gen.ts
│   │   │   │   ├── index.ts
│   │   │   │   ├── server.ts
│   │   │   │   └── v2
│   │   │   │       ├── client.ts
│   │   │   │       ├── gen
│   │   │   │       │   ├── client
│   │   │   │       │   │   ├── client.gen.ts
│   │   │   │       │   │   ├── index.ts
│   │   │   │       │   │   ├── types.gen.ts
│   │   │   │       │   │   └── utils.gen.ts
│   │   │   │       │   ├── client.gen.ts
│   │   │   │       │   ├── core
│   │   │   │       │   │   ├── auth.gen.ts
│   │   │   │       │   │   ├── bodySerializer.gen.ts
│   │   │   │       │   │   ├── params.gen.ts
│   │   │   │       │   │   ├── pathSerializer.gen.ts
│   │   │   │       │   │   ├── queryKeySerializer.gen.ts
│   │   │   │       │   │   ├── serverSentEvents.gen.ts
│   │   │   │       │   │   ├── types.gen.ts
│   │   │   │       │   │   └── utils.gen.ts
│   │   │   │       │   ├── sdk.gen.ts
│   │   │   │       │   └── types.gen.ts
│   │   │   │       ├── index.ts
│   │   │   │       └── server.ts
│   │   │   ├── sst-env.d.ts
│   │   │   └── tsconfig.json
│   │   └── openapi.json
│   ├── slack
│   │   ├── package.json
│   │   ├── README.md
│   │   ├── src
│   │   │   └── index.ts
│   │   ├── sst-env.d.ts
│   │   └── tsconfig.json
│   ├── ui
│   │   ├── package.json
│   │   ├── script
│   │   │   ├── colors.txt
│   │   │   └── tailwind.ts
│   │   ├── src
│   │   │   ├── assets
│   │   │   │   ├── audio
│   │   │   │   │   ├── alert-01.aac
│   │   │   │   │   ├── alert-02.aac
│   │   │   │   │   ├── alert-03.aac
│   │   │   │   │   ├── alert-04.aac
│   │   │   │   │   ├── alert-05.aac
│   │   │   │   │   ├── alert-06.aac
│   │   │   │   │   ├── alert-07.aac
│   │   │   │   │   ├── alert-08.aac
│   │   │   │   │   ├── alert-09.aac
│   │   │   │   │   ├── alert-10.aac
│   │   │   │   │   ├── bip-bop-01.aac
│   │   │   │   │   ├── bip-bop-02.aac
│   │   │   │   │   ├── bip-bop-03.aac
│   │   │   │   │   ├── bip-bop-04.aac
│   │   │   │   │   ├── bip-bop-05.aac
│   │   │   │   │   ├── bip-bop-06.aac
│   │   │   │   │   ├── bip-bop-07.aac
│   │   │   │   │   ├── bip-bop-08.aac
│   │   │   │   │   ├── bip-bop-09.aac
│   │   │   │   │   ├── bip-bop-10.aac
│   │   │   │   │   ├── nope-01.aac
│   │   │   │   │   ├── nope-02.aac
│   │   │   │   │   ├── nope-03.aac
│   │   │   │   │   ├── nope-04.aac
│   │   │   │   │   ├── nope-05.aac
│   │   │   │   │   ├── nope-06.aac
│   │   │   │   │   ├── nope-07.aac
│   │   │   │   │   ├── nope-08.aac
│   │   │   │   │   ├── nope-09.aac
│   │   │   │   │   ├── nope-10.aac
│   │   │   │   │   ├── nope-11.aac
│   │   │   │   │   ├── nope-12.aac
│   │   │   │   │   ├── staplebops-01.aac
│   │   │   │   │   ├── staplebops-02.aac
│   │   │   │   │   ├── staplebops-03.aac
│   │   │   │   │   ├── staplebops-04.aac
│   │   │   │   │   ├── staplebops-05.aac
│   │   │   │   │   ├── staplebops-06.aac
│   │   │   │   │   ├── staplebops-07.aac
│   │   │   │   │   ├── yup-01.aac
│   │   │   │   │   ├── yup-02.aac
│   │   │   │   │   ├── yup-03.aac
│   │   │   │   │   ├── yup-04.aac
│   │   │   │   │   ├── yup-05.aac
│   │   │   │   │   └── yup-06.aac
│   │   │   │   ├── favicon
│   │   │   │   │   ├── apple-touch-icon.png
│   │   │   │   │   ├── apple-touch-icon-v3.png
│   │   │   │   │   ├── favicon-96x96.png
│   │   │   │   │   ├── favicon-96x96-v3.png
│   │   │   │   │   ├── favicon.ico
│   │   │   │   │   ├── favicon.svg
│   │   │   │   │   ├── favicon-v3.ico
│   │   │   │   │   ├── favicon-v3.svg
│   │   │   │   │   ├── site.webmanifest
│   │   │   │   │   ├── web-app-manifest-192x192.png
│   │   │   │   │   └── web-app-manifest-512x512.png
│   │   │   │   ├── fonts
│   │   │   │   │   ├── BlexMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── BlexMonoNerdFontMono-Medium.woff2
│   │   │   │   │   ├── BlexMonoNerdFontMono-Regular.woff2
│   │   │   │   │   ├── cascadia-code-nerd-font-bold.woff2 -> CaskaydiaCoveNerdFontMono-Bold.woff2
│   │   │   │   │   ├── cascadia-code-nerd-font.woff2 -> CaskaydiaCoveNerdFontMono-Regular.woff2
│   │   │   │   │   ├── CaskaydiaCoveNerdFontMono-Bold.woff2
│   │   │   │   │   ├── CaskaydiaCoveNerdFontMono-Regular.woff2
│   │   │   │   │   ├── fira-code-nerd-font-bold.woff2 -> FiraCodeNerdFontMono-Bold.woff2
│   │   │   │   │   ├── FiraCodeNerdFontMono-Bold.woff2
│   │   │   │   │   ├── FiraCodeNerdFontMono-Regular.woff2
│   │   │   │   │   ├── fira-code-nerd-font.woff2 -> FiraCodeNerdFontMono-Regular.woff2
│   │   │   │   │   ├── geist-italic-medium.otf
│   │   │   │   │   ├── geist-italic-regular.otf
│   │   │   │   │   ├── geist-italic.ttf
│   │   │   │   │   ├── geist-italic.woff2
│   │   │   │   │   ├── geist-medium.otf
│   │   │   │   │   ├── geist-mono-bold.woff2 -> GeistMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── geist-mono-italic.ttf
│   │   │   │   │   ├── geist-mono-italic.woff2
│   │   │   │   │   ├── geist-mono-medium.woff2 -> GeistMonoNerdFontMono-Medium.woff2
│   │   │   │   │   ├── GeistMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── GeistMonoNerdFontMono-Medium.woff2
│   │   │   │   │   ├── GeistMonoNerdFontMono-Regular.woff2
│   │   │   │   │   ├── geist-mono.ttf
│   │   │   │   │   ├── geist-mono.woff2 -> GeistMonoNerdFontMono-Regular.woff2
│   │   │   │   │   ├── geist-regular.otf
│   │   │   │   │   ├── geist.ttf
│   │   │   │   │   ├── geist.woff2
│   │   │   │   │   ├── hack-nerd-font-bold.woff2 -> HackNerdFontMono-Bold.woff2
│   │   │   │   │   ├── HackNerdFontMono-Bold.woff2
│   │   │   │   │   ├── HackNerdFontMono-Regular.woff2
│   │   │   │   │   ├── hack-nerd-font.woff2 -> HackNerdFontMono-Regular.woff2
│   │   │   │   │   ├── ibm-plex-mono-bold.woff2 -> BlexMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── ibm-plex-mono-medium.woff2 -> BlexMonoNerdFontMono-Medium.woff2
│   │   │   │   │   ├── ibm-plex-mono.otf
│   │   │   │   │   ├── ibm-plex-mono.woff2 -> BlexMonoNerdFontMono-Regular.woff2
│   │   │   │   │   ├── inconsolata-nerd-font-bold.woff2 -> InconsolataNerdFontMono-Bold.woff2
│   │   │   │   │   ├── InconsolataNerdFontMono-Bold.woff2
│   │   │   │   │   ├── InconsolataNerdFontMono-Regular.woff2
│   │   │   │   │   ├── inconsolata-nerd-font.woff2 -> InconsolataNerdFontMono-Regular.woff2
│   │   │   │   │   ├── intel-one-mono-nerd-font-bold.woff2 -> IntoneMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── intel-one-mono-nerd-font.woff2 -> IntoneMonoNerdFontMono-Regular.woff2
│   │   │   │   │   ├── inter-italic.otf
│   │   │   │   │   ├── inter-italic.woff2
│   │   │   │   │   ├── inter.otf
│   │   │   │   │   ├── inter.woff2
│   │   │   │   │   ├── IntoneMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── IntoneMonoNerdFontMono-Regular.woff2
│   │   │   │   │   ├── iosevka-nerd-font-bold.woff2
│   │   │   │   │   ├── iosevka-nerd-font.woff2
│   │   │   │   │   ├── jetbrains-mono-nerd-font-bold.woff2 -> JetBrainsMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── JetBrainsMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── JetBrainsMonoNerdFontMono-Regular.woff2
│   │   │   │   │   ├── jetbrains-mono-nerd-font.woff2 -> JetBrainsMonoNerdFontMono-Regular.woff2
│   │   │   │   │   ├── meslo-lgs-nerd-font-bold.woff2 -> MesloLGSNerdFontMono-Bold.woff2
│   │   │   │   │   ├── MesloLGSNerdFontMono-Bold.woff2
│   │   │   │   │   ├── MesloLGSNerdFontMono-Regular.woff2
│   │   │   │   │   ├── meslo-lgs-nerd-font.woff2 -> MesloLGSNerdFontMono-Regular.woff2
│   │   │   │   │   ├── roboto-mono-nerd-font-bold.woff2 -> RobotoMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── RobotoMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── RobotoMonoNerdFontMono-Regular.woff2
│   │   │   │   │   ├── roboto-mono-nerd-font.woff2 -> RobotoMonoNerdFontMono-Regular.woff2
│   │   │   │   │   ├── SauceCodeProNerdFontMono-Bold.woff2
│   │   │   │   │   ├── SauceCodeProNerdFontMono-Regular.woff2
│   │   │   │   │   ├── source-code-pro-nerd-font-bold.woff2 -> SauceCodeProNerdFontMono-Bold.woff2
│   │   │   │   │   ├── source-code-pro-nerd-font.woff2 -> SauceCodeProNerdFontMono-Regular.woff2
│   │   │   │   │   ├── ubuntu-mono-nerd-font-bold.woff2 -> UbuntuMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── UbuntuMonoNerdFontMono-Bold.woff2
│   │   │   │   │   ├── UbuntuMonoNerdFontMono-Regular.woff2
│   │   │   │   │   └── ubuntu-mono-nerd-font.woff2 -> UbuntuMonoNerdFontMono-Regular.woff2
│   │   │   │   ├── icons
│   │   │   │   │   ├── app
│   │   │   │   │   │   ├── android-studio.svg
│   │   │   │   │   │   ├── antigravity.svg
│   │   │   │   │   │   ├── cursor.svg
│   │   │   │   │   │   ├── file-explorer.svg
│   │   │   │   │   │   ├── finder.png
│   │   │   │   │   │   ├── ghostty.svg
│   │   │   │   │   │   ├── iterm2.svg
│   │   │   │   │   │   ├── powershell.svg
│   │   │   │   │   │   ├── sublimetext.svg
│   │   │   │   │   │   ├── terminal.png
│   │   │   │   │   │   ├── textmate.png
│   │   │   │   │   │   ├── vscode.svg
│   │   │   │   │   │   ├── xcode.png
│   │   │   │   │   │   ├── zed-dark.svg
│   │   │   │   │   │   └── zed.svg
│   │   │   │   │   ├── file-types
│   │   │   │   │   │   ├── 3d.svg
│   │   │   │   │   │   ├── abap.svg
│   │   │   │   │   │   ├── abc.svg
│   │   │   │   │   │   ├── actionscript.svg
│   │   │   │   │   │   ├── ada.svg
│   │   │   │   │   │   ├── adobe-illustrator_light.svg
│   │   │   │   │   │   ├── adobe-illustrator.svg
│   │   │   │   │   │   ├── adobe-photoshop_light.svg
│   │   │   │   │   │   ├── adobe-photoshop.svg
│   │   │   │   │   │   ├── adobe-swc.svg
│   │   │   │   │   │   ├── adonis.svg
│   │   │   │   │   │   ├── advpl.svg
│   │   │   │   │   │   ├── amplify.svg
│   │   │   │   │   │   ├── android.svg
│   │   │   │   │   │   ├── angular.svg
│   │   │   │   │   │   ├── antlr.svg
│   │   │   │   │   │   ├── apiblueprint.svg
│   │   │   │   │   │   ├── apollo.svg
│   │   │   │   │   │   ├── applescript.svg
│   │   │   │   │   │   ├── apps-script.svg
│   │   │   │   │   │   ├── appveyor.svg
│   │   │   │   │   │   ├── architecture.svg
│   │   │   │   │   │   ├── arduino.svg
│   │   │   │   │   │   ├── asciidoc.svg
│   │   │   │   │   │   ├── assembly.svg
│   │   │   │   │   │   ├── astro-config.svg
│   │   │   │   │   │   ├── astro.svg
│   │   │   │   │   │   ├── astyle.svg
│   │   │   │   │   │   ├── audio.svg
│   │   │   │   │   │   ├── aurelia.svg
│   │   │   │   │   │   ├── authors.svg
│   │   │   │   │   │   ├── autohotkey.svg
│   │   │   │   │   │   ├── autoit.svg
│   │   │   │   │   │   ├── auto_light.svg
│   │   │   │   │   │   ├── auto.svg
│   │   │   │   │   │   ├── azure-pipelines.svg
│   │   │   │   │   │   ├── azure.svg
│   │   │   │   │   │   ├── babel.svg
│   │   │   │   │   │   ├── ballerina.svg
│   │   │   │   │   │   ├── bazel.svg
│   │   │   │   │   │   ├── bbx.svg
│   │   │   │   │   │   ├── beancount.svg
│   │   │   │   │   │   ├── bench-js.svg
│   │   │   │   │   │   ├── bench-jsx.svg
│   │   │   │   │   │   ├── bench-ts.svg
│   │   │   │   │   │   ├── bibliography.svg
│   │   │   │   │   │   ├── bibtex-style.svg
│   │   │   │   │   │   ├── bicep.svg
│   │   │   │   │   │   ├── biome.svg
│   │   │   │   │   │   ├── bitbucket.svg
│   │   │   │   │   │   ├── bithound.svg
│   │   │   │   │   │   ├── blender.svg
│   │   │   │   │   │   ├── blink_light.svg
│   │   │   │   │   │   ├── blink.svg
│   │   │   │   │   │   ├── blitz.svg
│   │   │   │   │   │   ├── bower.svg
│   │   │   │   │   │   ├── brainfuck.svg
│   │   │   │   │   │   ├── browserlist_light.svg
│   │   │   │   │   │   ├── browserlist.svg
│   │   │   │   │   │   ├── bruno.svg
│   │   │   │   │   │   ├── bucklescript.svg
│   │   │   │   │   │   ├── buck.svg
│   │   │   │   │   │   ├── buildkite.svg
│   │   │   │   │   │   ├── bun_light.svg
│   │   │   │   │   │   ├── bun.svg
│   │   │   │   │   │   ├── c3.svg
│   │   │   │   │   │   ├── cabal.svg
│   │   │   │   │   │   ├── caddy.svg
│   │   │   │   │   │   ├── cadence.svg
│   │   │   │   │   │   ├── cairo.svg
│   │   │   │   │   │   ├── cake.svg
│   │   │   │   │   │   ├── capacitor.svg
│   │   │   │   │   │   ├── capnp.svg
│   │   │   │   │   │   ├── cbx.svg
│   │   │   │   │   │   ├── cds.svg
│   │   │   │   │   │   ├── certificate.svg
│   │   │   │   │   │   ├── changelog.svg
│   │   │   │   │   │   ├── chess_light.svg
│   │   │   │   │   │   ├── chess.svg
│   │   │   │   │   │   ├── chrome.svg
│   │   │   │   │   │   ├── circleci_light.svg
│   │   │   │   │   │   ├── circleci.svg
│   │   │   │   │   │   ├── citation.svg
│   │   │   │   │   │   ├── clangd.svg
│   │   │   │   │   │   ├── claude.svg
│   │   │   │   │   │   ├── cline.svg
│   │   │   │   │   │   ├── clojure.svg
│   │   │   │   │   │   ├── cloudfoundry.svg
│   │   │   │   │   │   ├── cmake.svg
│   │   │   │   │   │   ├── coala.svg
│   │   │   │   │   │   ├── cobol.svg
│   │   │   │   │   │   ├── coconut.svg
│   │   │   │   │   │   ├── code-climate_light.svg
│   │   │   │   │   │   ├── code-climate.svg
│   │   │   │   │   │   ├── codecov.svg
│   │   │   │   │   │   ├── codeowners.svg
│   │   │   │   │   │   ├── coderabbit-ai.svg
│   │   │   │   │   │   ├── coffee.svg
│   │   │   │   │   │   ├── coldfusion.svg
│   │   │   │   │   │   ├── coloredpetrinets.svg
│   │   │   │   │   │   ├── command.svg
│   │   │   │   │   │   ├── commitizen.svg
│   │   │   │   │   │   ├── commitlint.svg
│   │   │   │   │   │   ├── concourse.svg
│   │   │   │   │   │   ├── conduct.svg
│   │   │   │   │   │   ├── console.svg
│   │   │   │   │   │   ├── contentlayer.svg
│   │   │   │   │   │   ├── context.svg
│   │   │   │   │   │   ├── contributing.svg
│   │   │   │   │   │   ├── controller.svg
│   │   │   │   │   │   ├── copilot_light.svg
│   │   │   │   │   │   ├── copilot.svg
│   │   │   │   │   │   ├── cpp.svg
│   │   │   │   │   │   ├── craco.svg
│   │   │   │   │   │   ├── credits.svg
│   │   │   │   │   │   ├── crystal_light.svg
│   │   │   │   │   │   ├── crystal.svg
│   │   │   │   │   │   ├── csharp.svg
│   │   │   │   │   │   ├── css-map.svg
│   │   │   │   │   │   ├── css.svg
│   │   │   │   │   │   ├── c.svg
│   │   │   │   │   │   ├── cucumber.svg
│   │   │   │   │   │   ├── cuda.svg
│   │   │   │   │   │   ├── cursor_light.svg
│   │   │   │   │   │   ├── cursor.svg
│   │   │   │   │   │   ├── cypress.svg
│   │   │   │   │   │   ├── dart_generated.svg
│   │   │   │   │   │   ├── dart.svg
│   │   │   │   │   │   ├── database.svg
│   │   │   │   │   │   ├── deepsource.svg
│   │   │   │   │   │   ├── denizenscript.svg
│   │   │   │   │   │   ├── deno_light.svg
│   │   │   │   │   │   ├── deno.svg
│   │   │   │   │   │   ├── dependabot.svg
│   │   │   │   │   │   ├── dependencies-update.svg
│   │   │   │   │   │   ├── dhall.svg
│   │   │   │   │   │   ├── diff.svg
│   │   │   │   │   │   ├── dinophp.svg
│   │   │   │   │   │   ├── disc.svg
│   │   │   │   │   │   ├── django.svg
│   │   │   │   │   │   ├── dll.svg
│   │   │   │   │   │   ├── docker.svg
│   │   │   │   │   │   ├── doctex-installer.svg
│   │   │   │   │   │   ├── document.svg
│   │   │   │   │   │   ├── dotjs.svg
│   │   │   │   │   │   ├── drawio.svg
│   │   │   │   │   │   ├── drizzle.svg
│   │   │   │   │   │   ├── drone_light.svg
│   │   │   │   │   │   ├── drone.svg
│   │   │   │   │   │   ├── d.svg
│   │   │   │   │   │   ├── duc.svg
│   │   │   │   │   │   ├── dune.svg
│   │   │   │   │   │   ├── edge.svg
│   │   │   │   │   │   ├── editorconfig.svg
│   │   │   │   │   │   ├── ejs.svg
│   │   │   │   │   │   ├── elixir.svg
│   │   │   │   │   │   ├── elm.svg
│   │   │   │   │   │   ├── email.svg
│   │   │   │   │   │   ├── ember.svg
│   │   │   │   │   │   ├── epub.svg
│   │   │   │   │   │   ├── erlang.svg
│   │   │   │   │   │   ├── esbuild.svg
│   │   │   │   │   │   ├── eslint.svg
│   │   │   │   │   │   ├── excalidraw.svg
│   │   │   │   │   │   ├── exe.svg
│   │   │   │   │   │   ├── fastlane.svg
│   │   │   │   │   │   ├── favicon.svg
│   │   │   │   │   │   ├── figma.svg
│   │   │   │   │   │   ├── firebase.svg
│   │   │   │   │   │   ├── flash.svg
│   │   │   │   │   │   ├── flow.svg
│   │   │   │   │   │   ├── folder-admin-open.svg
│   │   │   │   │   │   ├── folder-admin.svg
│   │   │   │   │   │   ├── folder-android-open.svg
│   │   │   │   │   │   ├── folder-android.svg
│   │   │   │   │   │   ├── folder-angular-open.svg
│   │   │   │   │   │   ├── folder-angular.svg
│   │   │   │   │   │   ├── folder-animation-open.svg
│   │   │   │   │   │   ├── folder-animation.svg
│   │   │   │   │   │   ├── folder-ansible-open.svg
│   │   │   │   │   │   ├── folder-ansible.svg
│   │   │   │   │   │   ├── folder-api-open.svg
│   │   │   │   │   │   ├── folder-api.svg
│   │   │   │   │   │   ├── folder-apollo-open.svg
│   │   │   │   │   │   ├── folder-apollo.svg
│   │   │   │   │   │   ├── folder-app-open.svg
│   │   │   │   │   │   ├── folder-app.svg
│   │   │   │   │   │   ├── folder-archive-open.svg
│   │   │   │   │   │   ├── folder-archive.svg
│   │   │   │   │   │   ├── folder-astro-open.svg
│   │   │   │   │   │   ├── folder-astro.svg
│   │   │   │   │   │   ├── folder-atom-open.svg
│   │   │   │   │   │   ├── folder-atom.svg
│   │   │   │   │   │   ├── folder-attachment-open.svg
│   │   │   │   │   │   ├── folder-attachment.svg
│   │   │   │   │   │   ├── folder-audio-open.svg
│   │   │   │   │   │   ├── folder-audio.svg
│   │   │   │   │   │   ├── folder-aurelia-open.svg
│   │   │   │   │   │   ├── folder-aurelia.svg
│   │   │   │   │   │   ├── folder-aws-open.svg
│   │   │   │   │   │   ├── folder-aws.svg
│   │   │   │   │   │   ├── folder-azure-pipelines-open.svg
│   │   │   │   │   │   ├── folder-azure-pipelines.svg
│   │   │   │   │   │   ├── folder-backup-open.svg
│   │   │   │   │   │   ├── folder-backup.svg
│   │   │   │   │   │   ├── folder-base-open.svg
│   │   │   │   │   │   ├── folder-base.svg
│   │   │   │   │   │   ├── folder-batch-open.svg
│   │   │   │   │   │   ├── folder-batch.svg
│   │   │   │   │   │   ├── folder-benchmark-open.svg
│   │   │   │   │   │   ├── folder-benchmark.svg
│   │   │   │   │   │   ├── folder-bibliography-open.svg
│   │   │   │   │   │   ├── folder-bibliography.svg
│   │   │   │   │   │   ├── folder-bicep-open.svg
│   │   │   │   │   │   ├── folder-bicep.svg
│   │   │   │   │   │   ├── folder-blender-open.svg
│   │   │   │   │   │   ├── folder-blender.svg
│   │   │   │   │   │   ├── folder-bloc-open.svg
│   │   │   │   │   │   ├── folder-bloc.svg
│   │   │   │   │   │   ├── folder-bower-open.svg
│   │   │   │   │   │   ├── folder-bower.svg
│   │   │   │   │   │   ├── folder-buildkite-open.svg
│   │   │   │   │   │   ├── folder-buildkite.svg
│   │   │   │   │   │   ├── folder-cart-open.svg
│   │   │   │   │   │   ├── folder-cart.svg
│   │   │   │   │   │   ├── folder-changesets-open.svg
│   │   │   │   │   │   ├── folder-changesets.svg
│   │   │   │   │   │   ├── folder-ci-open.svg
│   │   │   │   │   │   ├── folder-circleci-open.svg
│   │   │   │   │   │   ├── folder-circleci.svg
│   │   │   │   │   │   ├── folder-ci.svg
│   │   │   │   │   │   ├── folder-class-open.svg
│   │   │   │   │   │   ├── folder-class.svg
│   │   │   │   │   │   ├── folder-claude-open.svg
│   │   │   │   │   │   ├── folder-claude.svg
│   │   │   │   │   │   ├── folder-client-open.svg
│   │   │   │   │   │   ├── folder-client.svg
│   │   │   │   │   │   ├── folder-cline-open.svg
│   │   │   │   │   │   ├── folder-cline.svg
│   │   │   │   │   │   ├── folder-cloudflare-open.svg
│   │   │   │   │   │   ├── folder-cloudflare.svg
│   │   │   │   │   │   ├── folder-cloud-functions-open.svg
│   │   │   │   │   │   ├── folder-cloud-functions.svg
│   │   │   │   │   │   ├── folder-cluster-open.svg
│   │   │   │   │   │   ├── folder-cluster.svg
│   │   │   │   │   │   ├── folder-cobol-open.svg
│   │   │   │   │   │   ├── folder-cobol.svg
│   │   │   │   │   │   ├── folder-command-open.svg
│   │   │   │   │   │   ├── folder-command.svg
│   │   │   │   │   │   ├── folder-components-open.svg
│   │   │   │   │   │   ├── folder-components.svg
│   │   │   │   │   │   ├── folder-config-open.svg
│   │   │   │   │   │   ├── folder-config.svg
│   │   │   │   │   │   ├── folder-connection-open.svg
│   │   │   │   │   │   ├── folder-connection.svg
│   │   │   │   │   │   ├── folder-console-open.svg
│   │   │   │   │   │   ├── folder-console.svg
│   │   │   │   │   │   ├── folder-constant-open.svg
│   │   │   │   │   │   ├── folder-constant.svg
│   │   │   │   │   │   ├── folder-container-open.svg
│   │   │   │   │   │   ├── folder-container.svg
│   │   │   │   │   │   ├── folder-content-open.svg
│   │   │   │   │   │   ├── folder-content.svg
│   │   │   │   │   │   ├── folder-context-open.svg
│   │   │   │   │   │   ├── folder-context.svg
│   │   │   │   │   │   ├── folder-contract-open.svg
│   │   │   │   │   │   ├── folder-contract.svg
│   │   │   │   │   │   ├── folder-controller-open.svg
│   │   │   │   │   │   ├── folder-controller.svg
│   │   │   │   │   │   ├── folder-core-open.svg
│   │   │   │   │   │   ├── folder-core.svg
│   │   │   │   │   │   ├── folder-coverage-open.svg
│   │   │   │   │   │   ├── folder-coverage.svg
│   │   │   │   │   │   ├── folder-css-open.svg
│   │   │   │   │   │   ├── folder-css.svg
│   │   │   │   │   │   ├── folder-cursor_light.svg
│   │   │   │   │   │   ├── folder-cursor-open_light.svg
│   │   │   │   │   │   ├── folder-cursor-open.svg
│   │   │   │   │   │   ├── folder-cursor.svg
│   │   │   │   │   │   ├── folder-custom-open.svg
│   │   │   │   │   │   ├── folder-custom.svg
│   │   │   │   │   │   ├── folder-cypress-open.svg
│   │   │   │   │   │   ├── folder-cypress.svg
│   │   │   │   │   │   ├── folder-dart-open.svg
│   │   │   │   │   │   ├── folder-dart.svg
│   │   │   │   │   │   ├── folder-database-open.svg
│   │   │   │   │   │   ├── folder-database.svg
│   │   │   │   │   │   ├── folder-debug-open.svg
│   │   │   │   │   │   ├── folder-debug.svg
│   │   │   │   │   │   ├── folder-decorators-open.svg
│   │   │   │   │   │   ├── folder-decorators.svg
│   │   │   │   │   │   ├── folder-delta-open.svg
│   │   │   │   │   │   ├── folder-delta.svg
│   │   │   │   │   │   ├── folder-desktop-open.svg
│   │   │   │   │   │   ├── folder-desktop.svg
│   │   │   │   │   │   ├── folder-directive-open.svg
│   │   │   │   │   │   ├── folder-directive.svg
│   │   │   │   │   │   ├── folder-dist-open.svg
│   │   │   │   │   │   ├── folder-dist.svg
│   │   │   │   │   │   ├── folder-docker-open.svg
│   │   │   │   │   │   ├── folder-docker.svg
│   │   │   │   │   │   ├── folder-docs-open.svg
│   │   │   │   │   │   ├── folder-docs.svg
│   │   │   │   │   │   ├── folder-download-open.svg
│   │   │   │   │   │   ├── folder-download.svg
│   │   │   │   │   │   ├── folder-drizzle-open.svg
│   │   │   │   │   │   ├── folder-drizzle.svg
│   │   │   │   │   │   ├── folder-dump-open.svg
│   │   │   │   │   │   ├── folder-dump.svg
│   │   │   │   │   │   ├── folder-element-open.svg
│   │   │   │   │   │   ├── folder-element.svg
│   │   │   │   │   │   ├── folder-enum-open.svg
│   │   │   │   │   │   ├── folder-enum.svg
│   │   │   │   │   │   ├── folder-environment-open.svg
│   │   │   │   │   │   ├── folder-environment.svg
│   │   │   │   │   │   ├── folder-error-open.svg
│   │   │   │   │   │   ├── folder-error.svg
│   │   │   │   │   │   ├── folder-event-open.svg
│   │   │   │   │   │   ├── folder-event.svg
│   │   │   │   │   │   ├── folder-examples-open.svg
│   │   │   │   │   │   ├── folder-examples.svg
│   │   │   │   │   │   ├── folder-expo-open.svg
│   │   │   │   │   │   ├── folder-export-open.svg
│   │   │   │   │   │   ├── folder-export.svg
│   │   │   │   │   │   ├── folder-expo.svg
│   │   │   │   │   │   ├── folder-fastlane-open.svg
│   │   │   │   │   │   ├── folder-fastlane.svg
│   │   │   │   │   │   ├── folder-favicon-open.svg
│   │   │   │   │   │   ├── folder-favicon.svg
│   │   │   │   │   │   ├── folder-firebase-open.svg
│   │   │   │   │   │   ├── folder-firebase.svg
│   │   │   │   │   │   ├── folder-firestore-open.svg
│   │   │   │   │   │   ├── folder-firestore.svg
│   │   │   │   │   │   ├── folder-flow-open.svg
│   │   │   │   │   │   ├── folder-flow.svg
│   │   │   │   │   │   ├── folder-flutter-open.svg
│   │   │   │   │   │   ├── folder-flutter.svg
│   │   │   │   │   │   ├── folder-font-open.svg
│   │   │   │   │   │   ├── folder-font.svg
│   │   │   │   │   │   ├── folder-forgejo-open.svg
│   │   │   │   │   │   ├── folder-forgejo.svg
│   │   │   │   │   │   ├── folder-functions-open.svg
│   │   │   │   │   │   ├── folder-functions.svg
│   │   │   │   │   │   ├── folder-gamemaker-open.svg
│   │   │   │   │   │   ├── folder-gamemaker.svg
│   │   │   │   │   │   ├── folder-generator-open.svg
│   │   │   │   │   │   ├── folder-generator.svg
│   │   │   │   │   │   ├── folder-gh-workflows-open.svg
│   │   │   │   │   │   ├── folder-gh-workflows.svg
│   │   │   │   │   │   ├── folder-gitea-open.svg
│   │   │   │   │   │   ├── folder-gitea.svg
│   │   │   │   │   │   ├── folder-github-open.svg
│   │   │   │   │   │   ├── folder-github.svg
│   │   │   │   │   │   ├── folder-gitlab-open.svg
│   │   │   │   │   │   ├── folder-gitlab.svg
│   │   │   │   │   │   ├── folder-git-open.svg
│   │   │   │   │   │   ├── folder-git.svg
│   │   │   │   │   │   ├── folder-global-open.svg
│   │   │   │   │   │   ├── folder-global.svg
│   │   │   │   │   │   ├── folder-godot-open.svg
│   │   │   │   │   │   ├── folder-godot.svg
│   │   │   │   │   │   ├── folder-gradle-open.svg
│   │   │   │   │   │   ├── folder-gradle.svg
│   │   │   │   │   │   ├── folder-graphql-open.svg
│   │   │   │   │   │   ├── folder-graphql.svg
│   │   │   │   │   │   ├── folder-guard-open.svg
│   │   │   │   │   │   ├── folder-guard.svg
│   │   │   │   │   │   ├── folder-gulp-open.svg
│   │   │   │   │   │   ├── folder-gulp.svg
│   │   │   │   │   │   ├── folder-helm-open.svg
│   │   │   │   │   │   ├── folder-helm.svg
│   │   │   │   │   │   ├── folder-helper-open.svg
│   │   │   │   │   │   ├── folder-helper.svg
│   │   │   │   │   │   ├── folder-home-open.svg
│   │   │   │   │   │   ├── folder-home.svg
│   │   │   │   │   │   ├── folder-hook-open.svg
│   │   │   │   │   │   ├── folder-hook.svg
│   │   │   │   │   │   ├── folder-husky-open.svg
│   │   │   │   │   │   ├── folder-husky.svg
│   │   │   │   │   │   ├── folder-i18n-open.svg
│   │   │   │   │   │   ├── folder-i18n.svg
│   │   │   │   │   │   ├── folder-images-open.svg
│   │   │   │   │   │   ├── folder-images.svg
│   │   │   │   │   │   ├── folder-import-open.svg
│   │   │   │   │   │   ├── folder-import.svg
│   │   │   │   │   │   ├── folder-include-open.svg
│   │   │   │   │   │   ├── folder-include.svg
│   │   │   │   │   │   ├── folder-intellij_light.svg
│   │   │   │   │   │   ├── folder-intellij-open_light.svg
│   │   │   │   │   │   ├── folder-intellij-open.svg
│   │   │   │   │   │   ├── folder-intellij.svg
│   │   │   │   │   │   ├── folder-interceptor-open.svg
│   │   │   │   │   │   ├── folder-interceptor.svg
│   │   │   │   │   │   ├── folder-interface-open.svg
│   │   │   │   │   │   ├── folder-interface.svg
│   │   │   │   │   │   ├── folder-ios-open.svg
│   │   │   │   │   │   ├── folder-ios.svg
│   │   │   │   │   │   ├── folder-java-open.svg
│   │   │   │   │   │   ├── folder-javascript-open.svg
│   │   │   │   │   │   ├── folder-javascript.svg
│   │   │   │   │   │   ├── folder-java.svg
│   │   │   │   │   │   ├── folder-jinja_light.svg
│   │   │   │   │   │   ├── folder-jinja-open_light.svg
│   │   │   │   │   │   ├── folder-jinja-open.svg
│   │   │   │   │   │   ├── folder-jinja.svg
│   │   │   │   │   │   ├── folder-job-open.svg
│   │   │   │   │   │   ├── folder-job.svg
│   │   │   │   │   │   ├── folder-json-open.svg
│   │   │   │   │   │   ├── folder-json.svg
│   │   │   │   │   │   ├── folder-jupyter-open.svg
│   │   │   │   │   │   ├── folder-jupyter.svg
│   │   │   │   │   │   ├── folder-keys-open.svg
│   │   │   │   │   │   ├── folder-keys.svg
│   │   │   │   │   │   ├── folder-kubernetes-open.svg
│   │   │   │   │   │   ├── folder-kubernetes.svg
│   │   │   │   │   │   ├── folder-kusto-open.svg
│   │   │   │   │   │   ├── folder-kusto.svg
│   │   │   │   │   │   ├── folder-layout-open.svg
│   │   │   │   │   │   ├── folder-layout.svg
│   │   │   │   │   │   ├── folder-lefthook-open.svg
│   │   │   │   │   │   ├── folder-lefthook.svg
│   │   │   │   │   │   ├── folder-less-open.svg
│   │   │   │   │   │   ├── folder-less.svg
│   │   │   │   │   │   ├── folder-lib-open.svg
│   │   │   │   │   │   ├── folder-lib.svg
│   │   │   │   │   │   ├── folder-link-open.svg
│   │   │   │   │   │   ├── folder-link.svg
│   │   │   │   │   │   ├── folder-linux-open.svg
│   │   │   │   │   │   ├── folder-linux.svg
│   │   │   │   │   │   ├── folder-liquibase-open.svg
│   │   │   │   │   │   ├── folder-liquibase.svg
│   │   │   │   │   │   ├── folder-log-open.svg
│   │   │   │   │   │   ├── folder-log.svg
│   │   │   │   │   │   ├── folder-lottie-open.svg
│   │   │   │   │   │   ├── folder-lottie.svg
│   │   │   │   │   │   ├── folder-lua-open.svg
│   │   │   │   │   │   ├── folder-lua.svg
│   │   │   │   │   │   ├── folder-luau-open.svg
│   │   │   │   │   │   ├── folder-luau.svg
│   │   │   │   │   │   ├── folder-macos-open.svg
│   │   │   │   │   │   ├── folder-macos.svg
│   │   │   │   │   │   ├── folder-mail-open.svg
│   │   │   │   │   │   ├── folder-mail.svg
│   │   │   │   │   │   ├── folder-mappings-open.svg
│   │   │   │   │   │   ├── folder-mappings.svg
│   │   │   │   │   │   ├── folder-markdown-open.svg
│   │   │   │   │   │   ├── folder-markdown.svg
│   │   │   │   │   │   ├── folder-mercurial-open.svg
│   │   │   │   │   │   ├── folder-mercurial.svg
│   │   │   │   │   │   ├── folder-messages-open.svg
│   │   │   │   │   │   ├── folder-messages.svg
│   │   │   │   │   │   ├── folder-meta-open.svg
│   │   │   │   │   │   ├── folder-meta.svg
│   │   │   │   │   │   ├── folder-middleware-open.svg
│   │   │   │   │   │   ├── folder-middleware.svg
│   │   │   │   │   │   ├── folder-mjml-open.svg
│   │   │   │   │   │   ├── folder-mjml.svg
│   │   │   │   │   │   ├── folder-mobile-open.svg
│   │   │   │   │   │   ├── folder-mobile.svg
│   │   │   │   │   │   ├── folder-mock-open.svg
│   │   │   │   │   │   ├── folder-mock.svg
│   │   │   │   │   │   ├── folder-mojo-open.svg
│   │   │   │   │   │   ├── folder-mojo.svg
│   │   │   │   │   │   ├── folder-molecule-open.svg
│   │   │   │   │   │   ├── folder-molecule.svg
│   │   │   │   │   │   ├── folder-moon-open.svg
│   │   │   │   │   │   ├── folder-moon.svg
│   │   │   │   │   │   ├── folder-netlify-open.svg
│   │   │   │   │   │   ├── folder-netlify.svg
│   │   │   │   │   │   ├── folder-next-open.svg
│   │   │   │   │   │   ├── folder-next.svg
│   │   │   │   │   │   ├── folder-ngrx-store-open.svg
│   │   │   │   │   │   ├── folder-ngrx-store.svg
│   │   │   │   │   │   ├── folder-node-open.svg
│   │   │   │   │   │   ├── folder-node.svg
│   │   │   │   │   │   ├── folder-nuxt-open.svg
│   │   │   │   │   │   ├── folder-nuxt.svg
│   │   │   │   │   │   ├── folder-obsidian-open.svg
│   │   │   │   │   │   ├── folder-obsidian.svg
│   │   │   │   │   │   ├── folder-open.svg
│   │   │   │   │   │   ├── folder-organism-open.svg
│   │   │   │   │   │   ├── folder-organism.svg
│   │   │   │   │   │   ├── folder-other-open.svg
│   │   │   │   │   │   ├── folder-other.svg
│   │   │   │   │   │   ├── folder-packages-open.svg
│   │   │   │   │   │   ├── folder-packages.svg
│   │   │   │   │   │   ├── folder-pdf-open.svg
│   │   │   │   │   │   ├── folder-pdf.svg
│   │   │   │   │   │   ├── folder-pdm-open.svg
│   │   │   │   │   │   ├── folder-pdm.svg
│   │   │   │   │   │   ├── folder-phpmailer-open.svg
│   │   │   │   │   │   ├── folder-phpmailer.svg
│   │   │   │   │   │   ├── folder-php-open.svg
│   │   │   │   │   │   ├── folder-php.svg
│   │   │   │   │   │   ├── folder-pipe-open.svg
│   │   │   │   │   │   ├── folder-pipe.svg
│   │   │   │   │   │   ├── folder-plastic-open.svg
│   │   │   │   │   │   ├── folder-plastic.svg
│   │   │   │   │   │   ├── folder-plugin-open.svg
│   │   │   │   │   │   ├── folder-plugin.svg
│   │   │   │   │   │   ├── folder-policy-open.svg
│   │   │   │   │   │   ├── folder-policy.svg
│   │   │   │   │   │   ├── folder-powershell-open.svg
│   │   │   │   │   │   ├── folder-powershell.svg
│   │   │   │   │   │   ├── folder-prisma-open.svg
│   │   │   │   │   │   ├── folder-prisma.svg
│   │   │   │   │   │   ├── folder-private-open.svg
│   │   │   │   │   │   ├── folder-private.svg
│   │   │   │   │   │   ├── folder-project-open.svg
│   │   │   │   │   │   ├── folder-project.svg
│   │   │   │   │   │   ├── folder-prompts-open.svg
│   │   │   │   │   │   ├── folder-prompts.svg
│   │   │   │   │   │   ├── folder-proto-open.svg
│   │   │   │   │   │   ├── folder-proto.svg
│   │   │   │   │   │   ├── folder-public-open.svg
│   │   │   │   │   │   ├── folder-public.svg
│   │   │   │   │   │   ├── folder-python-open.svg
│   │   │   │   │   │   ├── folder-python.svg
│   │   │   │   │   │   ├── folder-pytorch-open.svg
│   │   │   │   │   │   ├── folder-pytorch.svg
│   │   │   │   │   │   ├── folder-quasar-open.svg
│   │   │   │   │   │   ├── folder-quasar.svg
│   │   │   │   │   │   ├── folder-queue-open.svg
│   │   │   │   │   │   ├── folder-queue.svg
│   │   │   │   │   │   ├── folder-react-components-open.svg
│   │   │   │   │   │   ├── folder-react-components.svg
│   │   │   │   │   │   ├── folder-redux-reducer-open.svg
│   │   │   │   │   │   ├── folder-redux-reducer.svg
│   │   │   │   │   │   ├── folder-repository-open.svg
│   │   │   │   │   │   ├── folder-repository.svg
│   │   │   │   │   │   ├── folder-resolver-open.svg
│   │   │   │   │   │   ├── folder-resolver.svg
│   │   │   │   │   │   ├── folder-resource-open.svg
│   │   │   │   │   │   ├── folder-resource.svg
│   │   │   │   │   │   ├── folder-review-open.svg
│   │   │   │   │   │   ├── folder-review.svg
│   │   │   │   │   │   ├── folder-robot-open.svg
│   │   │   │   │   │   ├── folder-robot.svg
│   │   │   │   │   │   ├── folder-routes-open.svg
│   │   │   │   │   │   ├── folder-routes.svg
│   │   │   │   │   │   ├── folder-rules-open.svg
│   │   │   │   │   │   ├── folder-rules.svg
│   │   │   │   │   │   ├── folder-rust-open.svg
│   │   │   │   │   │   ├── folder-rust.svg
│   │   │   │   │   │   ├── folder-sandbox-open.svg
│   │   │   │   │   │   ├── folder-sandbox.svg
│   │   │   │   │   │   ├── folder-sass-open.svg
│   │   │   │   │   │   ├── folder-sass.svg
│   │   │   │   │   │   ├── folder-scala-open.svg
│   │   │   │   │   │   ├── folder-scala.svg
│   │   │   │   │   │   ├── folder-scons-open.svg
│   │   │   │   │   │   ├── folder-scons.svg
│   │   │   │   │   │   ├── folder-scripts-open.svg
│   │   │   │   │   │   ├── folder-scripts.svg
│   │   │   │   │   │   ├── folder-secure-open.svg
│   │   │   │   │   │   ├── folder-secure.svg
│   │   │   │   │   │   ├── folder-seeders-open.svg
│   │   │   │   │   │   ├── folder-seeders.svg
│   │   │   │   │   │   ├── folder-serverless-open.svg
│   │   │   │   │   │   ├── folder-serverless.svg
│   │   │   │   │   │   ├── folder-server-open.svg
│   │   │   │   │   │   ├── folder-server.svg
│   │   │   │   │   │   ├── folder-shader-open.svg
│   │   │   │   │   │   ├── folder-shader.svg
│   │   │   │   │   │   ├── folder-shared-open.svg
│   │   │   │   │   │   ├── folder-shared.svg
│   │   │   │   │   │   ├── folder-snapcraft-open.svg
│   │   │   │   │   │   ├── folder-snapcraft.svg
│   │   │   │   │   │   ├── folder-snippet-open.svg
│   │   │   │   │   │   ├── folder-snippet.svg
│   │   │   │   │   │   ├── folder-src-open.svg
│   │   │   │   │   │   ├── folder-src.svg
│   │   │   │   │   │   ├── folder-src-tauri-open.svg
│   │   │   │   │   │   ├── folder-src-tauri.svg
│   │   │   │   │   │   ├── folder-stack-open.svg
│   │   │   │   │   │   ├── folder-stack.svg
│   │   │   │   │   │   ├── folder-stencil-open.svg
│   │   │   │   │   │   ├── folder-stencil.svg
│   │   │   │   │   │   ├── folder-store-open.svg
│   │   │   │   │   │   ├── folder-store.svg
│   │   │   │   │   │   ├── folder-storybook-open.svg
│   │   │   │   │   │   ├── folder-storybook.svg
│   │   │   │   │   │   ├── folder-stylus-open.svg
│   │   │   │   │   │   ├── folder-stylus.svg
│   │   │   │   │   │   ├── folder-sublime-open.svg
│   │   │   │   │   │   ├── folder-sublime.svg
│   │   │   │   │   │   ├── folder-supabase-open.svg
│   │   │   │   │   │   ├── folder-supabase.svg
│   │   │   │   │   │   ├── folder-svelte-open.svg
│   │   │   │   │   │   ├── folder-svelte.svg
│   │   │   │   │   │   ├── folder.svg
│   │   │   │   │   │   ├── folder-svg-open.svg
│   │   │   │   │   │   ├── folder-svg.svg
│   │   │   │   │   │   ├── folder-syntax-open.svg
│   │   │   │   │   │   ├── folder-syntax.svg
│   │   │   │   │   │   ├── folder-target-open.svg
│   │   │   │   │   │   ├── folder-target.svg
│   │   │   │   │   │   ├── folder-taskfile-open.svg
│   │   │   │   │   │   ├── folder-taskfile.svg
│   │   │   │   │   │   ├── folder-tasks-open.svg
│   │   │   │   │   │   ├── folder-tasks.svg
│   │   │   │   │   │   ├── folder-television-open.svg
│   │   │   │   │   │   ├── folder-television.svg
│   │   │   │   │   │   ├── folder-template-open.svg
│   │   │   │   │   │   ├── folder-template.svg
│   │   │   │   │   │   ├── folder-temp-open.svg
│   │   │   │   │   │   ├── folder-temp.svg
│   │   │   │   │   │   ├── folder-terraform-open.svg
│   │   │   │   │   │   ├── folder-terraform.svg
│   │   │   │   │   │   ├── folder-test-open.svg
│   │   │   │   │   │   ├── folder-test.svg
│   │   │   │   │   │   ├── folder-theme-open.svg
│   │   │   │   │   │   ├── folder-theme.svg
│   │   │   │   │   │   ├── folder-tools-open.svg
│   │   │   │   │   │   ├── folder-tools.svg
│   │   │   │   │   │   ├── folder-trash-open.svg
│   │   │   │   │   │   ├── folder-trash.svg
│   │   │   │   │   │   ├── folder-trigger-open.svg
│   │   │   │   │   │   ├── folder-trigger.svg
│   │   │   │   │   │   ├── folder-turborepo-open.svg
│   │   │   │   │   │   ├── folder-turborepo.svg
│   │   │   │   │   │   ├── folder-typescript-open.svg
│   │   │   │   │   │   ├── folder-typescript.svg
│   │   │   │   │   │   ├── folder-ui-open.svg
│   │   │   │   │   │   ├── folder-ui.svg
│   │   │   │   │   │   ├── folder-unity-open.svg
│   │   │   │   │   │   ├── folder-unity.svg
│   │   │   │   │   │   ├── folder-update-open.svg
│   │   │   │   │   │   ├── folder-update.svg
│   │   │   │   │   │   ├── folder-upload-open.svg
│   │   │   │   │   │   ├── folder-upload.svg
│   │   │   │   │   │   ├── folder-utils-open.svg
│   │   │   │   │   │   ├── folder-utils.svg
│   │   │   │   │   │   ├── folder-vercel-open.svg
│   │   │   │   │   │   ├── folder-vercel.svg
│   │   │   │   │   │   ├── folder-verdaccio-open.svg
│   │   │   │   │   │   ├── folder-verdaccio.svg
│   │   │   │   │   │   ├── folder-video-open.svg
│   │   │   │   │   │   ├── folder-video.svg
│   │   │   │   │   │   ├── folder-views-open.svg
│   │   │   │   │   │   ├── folder-views.svg
│   │   │   │   │   │   ├── folder-vm-open.svg
│   │   │   │   │   │   ├── folder-vm.svg
│   │   │   │   │   │   ├── folder-vscode-open.svg
│   │   │   │   │   │   ├── folder-vscode.svg
│   │   │   │   │   │   ├── folder-vue-directives-open.svg
│   │   │   │   │   │   ├── folder-vue-directives.svg
│   │   │   │   │   │   ├── folder-vue-open.svg
│   │   │   │   │   │   ├── folder-vuepress-open.svg
│   │   │   │   │   │   ├── folder-vuepress.svg
│   │   │   │   │   │   ├── folder-vue.svg
│   │   │   │   │   │   ├── folder-vuex-store-open.svg
│   │   │   │   │   │   ├── folder-vuex-store.svg
│   │   │   │   │   │   ├── folder-wakatime-open.svg
│   │   │   │   │   │   ├── folder-wakatime.svg
│   │   │   │   │   │   ├── folder-webpack-open.svg
│   │   │   │   │   │   ├── folder-webpack.svg
│   │   │   │   │   │   ├── folder-windows-open.svg
│   │   │   │   │   │   ├── folder-windows.svg
│   │   │   │   │   │   ├── folder-wordpress-open.svg
│   │   │   │   │   │   ├── folder-wordpress.svg
│   │   │   │   │   │   ├── folder-yarn-open.svg
│   │   │   │   │   │   ├── folder-yarn.svg
│   │   │   │   │   │   ├── folder-zeabur-open.svg
│   │   │   │   │   │   ├── folder-zeabur.svg
│   │   │   │   │   │   ├── font.svg
│   │   │   │   │   │   ├── forth.svg
│   │   │   │   │   │   ├── fortran.svg
│   │   │   │   │   │   ├── foxpro.svg
│   │   │   │   │   │   ├── freemarker.svg
│   │   │   │   │   │   ├── fsharp.svg
│   │   │   │   │   │   ├── fusebox.svg
│   │   │   │   │   │   ├── gamemaker.svg
│   │   │   │   │   │   ├── garden.svg
│   │   │   │   │   │   ├── gatsby.svg
│   │   │   │   │   │   ├── gcp.svg
│   │   │   │   │   │   ├── gemfile.svg
│   │   │   │   │   │   ├── gemini-ai.svg
│   │   │   │   │   │   ├── gemini.svg
│   │   │   │   │   │   ├── github-actions-workflow.svg
│   │   │   │   │   │   ├── github-sponsors.svg
│   │   │   │   │   │   ├── gitlab.svg
│   │   │   │   │   │   ├── gitpod.svg
│   │   │   │   │   │   ├── git.svg
│   │   │   │   │   │   ├── gleam.svg
│   │   │   │   │   │   ├── gnuplot.svg
│   │   │   │   │   │   ├── godot-assets.svg
│   │   │   │   │   │   ├── godot.svg
│   │   │   │   │   │   ├── go_gopher.svg
│   │   │   │   │   │   ├── go-mod.svg
│   │   │   │   │   │   ├── go.svg
│   │   │   │   │   │   ├── gradle.svg
│   │   │   │   │   │   ├── grafana-alloy.svg
│   │   │   │   │   │   ├── grain.svg
│   │   │   │   │   │   ├── graphcool.svg
│   │   │   │   │   │   ├── graphql.svg
│   │   │   │   │   │   ├── gridsome.svg
│   │   │   │   │   │   ├── groovy.svg
│   │   │   │   │   │   ├── grunt.svg
│   │   │   │   │   │   ├── gulp.svg
│   │   │   │   │   │   ├── hack.svg
│   │   │   │   │   │   ├── hadolint.svg
│   │   │   │   │   │   ├── haml.svg
│   │   │   │   │   │   ├── handlebars.svg
│   │   │   │   │   │   ├── hardhat.svg
│   │   │   │   │   │   ├── harmonix.svg
│   │   │   │   │   │   ├── haskell.svg
│   │   │   │   │   │   ├── haxe.svg
│   │   │   │   │   │   ├── hcl_light.svg
│   │   │   │   │   │   ├── hcl.svg
│   │   │   │   │   │   ├── helm.svg
│   │   │   │   │   │   ├── heroku.svg
│   │   │   │   │   │   ├── hex.svg
│   │   │   │   │   │   ├── histoire.svg
│   │   │   │   │   │   ├── hjson.svg
│   │   │   │   │   │   ├── horusec.svg
│   │   │   │   │   │   ├── hosts_light.svg
│   │   │   │   │   │   ├── hosts.svg
│   │   │   │   │   │   ├── hpp.svg
│   │   │   │   │   │   ├── h.svg
│   │   │   │   │   │   ├── html.svg
│   │   │   │   │   │   ├── http.svg
│   │   │   │   │   │   ├── huff_light.svg
│   │   │   │   │   │   ├── huff.svg
│   │   │   │   │   │   ├── hurl.svg
│   │   │   │   │   │   ├── husky.svg
│   │   │   │   │   │   ├── i18n.svg
│   │   │   │   │   │   ├── idris.svg
│   │   │   │   │   │   ├── ifanr-cloud.svg
│   │   │   │   │   │   ├── image.svg
│   │   │   │   │   │   ├── imba.svg
│   │   │   │   │   │   ├── installation.svg
│   │   │   │   │   │   ├── ionic.svg
│   │   │   │   │   │   ├── istanbul.svg
│   │   │   │   │   │   ├── jar.svg
│   │   │   │   │   │   ├── javaclass.svg
│   │   │   │   │   │   ├── javascript-map.svg
│   │   │   │   │   │   ├── javascript.svg
│   │   │   │   │   │   ├── java.svg
│   │   │   │   │   │   ├── jenkins.svg
│   │   │   │   │   │   ├── jest.svg
│   │   │   │   │   │   ├── jinja_light.svg
│   │   │   │   │   │   ├── jinja.svg
│   │   │   │   │   │   ├── jsconfig.svg
│   │   │   │   │   │   ├── json.svg
│   │   │   │   │   │   ├── jsr_light.svg
│   │   │   │   │   │   ├── jsr.svg
│   │   │   │   │   │   ├── julia.svg
│   │   │   │   │   │   ├── jupyter.svg
│   │   │   │   │   │   ├── just.svg
│   │   │   │   │   │   ├── karma.svg
│   │   │   │   │   │   ├── kcl.svg
│   │   │   │   │   │   ├── keystatic.svg
│   │   │   │   │   │   ├── key.svg
│   │   │   │   │   │   ├── kivy.svg
│   │   │   │   │   │   ├── kl.svg
│   │   │   │   │   │   ├── knip.svg
│   │   │   │   │   │   ├── kotlin.svg
│   │   │   │   │   │   ├── kubernetes.svg
│   │   │   │   │   │   ├── kusto.svg
│   │   │   │   │   │   ├── label.svg
│   │   │   │   │   │   ├── laravel.svg
│   │   │   │   │   │   ├── latexmk.svg
│   │   │   │   │   │   ├── lbx.svg
│   │   │   │   │   │   ├── lefthook.svg
│   │   │   │   │   │   ├── lerna.svg
│   │   │   │   │   │   ├── less.svg
│   │   │   │   │   │   ├── liara.svg
│   │   │   │   │   │   ├── lib.svg
│   │   │   │   │   │   ├── lighthouse.svg
│   │   │   │   │   │   ├── lilypond.svg
│   │   │   │   │   │   ├── lintstaged.svg
│   │   │   │   │   │   ├── liquid.svg
│   │   │   │   │   │   ├── lisp.svg
│   │   │   │   │   │   ├── livescript.svg
│   │   │   │   │   │   ├── lock.svg
│   │   │   │   │   │   ├── log.svg
│   │   │   │   │   │   ├── lolcode.svg
│   │   │   │   │   │   ├── lottie.svg
│   │   │   │   │   │   ├── lua.svg
│   │   │   │   │   │   ├── luau.svg
│   │   │   │   │   │   ├── lyric.svg
│   │   │   │   │   │   ├── makefile.svg
│   │   │   │   │   │   ├── markdoc-config.svg
│   │   │   │   │   │   ├── markdoc.svg
│   │   │   │   │   │   ├── markdownlint.svg
│   │   │   │   │   │   ├── markdown.svg
│   │   │   │   │   │   ├── markojs.svg
│   │   │   │   │   │   ├── mathematica.svg
│   │   │   │   │   │   ├── matlab.svg
│   │   │   │   │   │   ├── maven.svg
│   │   │   │   │   │   ├── mdsvex.svg
│   │   │   │   │   │   ├── mdx.svg
│   │   │   │   │   │   ├── mercurial.svg
│   │   │   │   │   │   ├── merlin.svg
│   │   │   │   │   │   ├── mermaid.svg
│   │   │   │   │   │   ├── meson.svg
│   │   │   │   │   │   ├── minecraft-fabric.svg
│   │   │   │   │   │   ├── minecraft.svg
│   │   │   │   │   │   ├── mint.svg
│   │   │   │   │   │   ├── mjml.svg
│   │   │   │   │   │   ├── mocha.svg
│   │   │   │   │   │   ├── modernizr.svg
│   │   │   │   │   │   ├── mojo.svg
│   │   │   │   │   │   ├── moonscript.svg
│   │   │   │   │   │   ├── moon.svg
│   │   │   │   │   │   ├── mxml.svg
│   │   │   │   │   │   ├── nano-staged_light.svg
│   │   │   │   │   │   ├── nano-staged.svg
│   │   │   │   │   │   ├── ndst.svg
│   │   │   │   │   │   ├── nest.svg
│   │   │   │   │   │   ├── netlify_light.svg
│   │   │   │   │   │   ├── netlify.svg
│   │   │   │   │   │   ├── next_light.svg
│   │   │   │   │   │   ├── next.svg
│   │   │   │   │   │   ├── nginx.svg
│   │   │   │   │   │   ├── ngrx-actions.svg
│   │   │   │   │   │   ├── ngrx-effects.svg
│   │   │   │   │   │   ├── ngrx-entity.svg
│   │   │   │   │   │   ├── ngrx-reducer.svg
│   │   │   │   │   │   ├── ngrx-selectors.svg
│   │   │   │   │   │   ├── ngrx-state.svg
│   │   │   │   │   │   ├── nim.svg
│   │   │   │   │   │   ├── nix.svg
│   │   │   │   │   │   ├── nodejs_alt.svg
│   │   │   │   │   │   ├── nodejs.svg
│   │   │   │   │   │   ├── nodemon.svg
│   │   │   │   │   │   ├── npm.svg
│   │   │   │   │   │   ├── nuget.svg
│   │   │   │   │   │   ├── nunjucks.svg
│   │   │   │   │   │   ├── nuxt.svg
│   │   │   │   │   │   ├── nx.svg
│   │   │   │   │   │   ├── objective-cpp.svg
│   │   │   │   │   │   ├── objective-c.svg
│   │   │   │   │   │   ├── ocaml.svg
│   │   │   │   │   │   ├── odin.svg
│   │   │   │   │   │   ├── opam.svg
│   │   │   │   │   │   ├── opa.svg
│   │   │   │   │   │   ├── openapi_light.svg
│   │   │   │   │   │   ├── openapi.svg
│   │   │   │   │   │   ├── otne.svg
│   │   │   │   │   │   ├── oxlint.svg
│   │   │   │   │   │   ├── packship.svg
│   │   │   │   │   │   ├── palette.svg
│   │   │   │   │   │   ├── panda.svg
│   │   │   │   │   │   ├── parcel.svg
│   │   │   │   │   │   ├── pascal.svg
│   │   │   │   │   │   ├── pawn.svg
│   │   │   │   │   │   ├── payload_light.svg
│   │   │   │   │   │   ├── payload.svg
│   │   │   │   │   │   ├── pdf.svg
│   │   │   │   │   │   ├── pdm.svg
│   │   │   │   │   │   ├── percy.svg
│   │   │   │   │   │   ├── perl.svg
│   │   │   │   │   │   ├── php-cs-fixer.svg
│   │   │   │   │   │   ├── php_elephant_pink.svg
│   │   │   │   │   │   ├── php_elephant.svg
│   │   │   │   │   │   ├── phpstan.svg
│   │   │   │   │   │   ├── php.svg
│   │   │   │   │   │   ├── phpunit.svg
│   │   │   │   │   │   ├── pinejs.svg
│   │   │   │   │   │   ├── pipeline.svg
│   │   │   │   │   │   ├── pkl.svg
│   │   │   │   │   │   ├── plastic.svg
│   │   │   │   │   │   ├── playwright.svg
│   │   │   │   │   │   ├── plop.svg
│   │   │   │   │   │   ├── pm2-ecosystem.svg
│   │   │   │   │   │   ├── pnpm_light.svg
│   │   │   │   │   │   ├── pnpm.svg
│   │   │   │   │   │   ├── poetry.svg
│   │   │   │   │   │   ├── postcss.svg
│   │   │   │   │   │   ├── posthtml.svg
│   │   │   │   │   │   ├── powerpoint.svg
│   │   │   │   │   │   ├── powershell.svg
│   │   │   │   │   │   ├── pre-commit.svg
│   │   │   │   │   │   ├── prettier.svg
│   │   │   │   │   │   ├── prisma.svg
│   │   │   │   │   │   ├── processing.svg
│   │   │   │   │   │   ├── prolog.svg
│   │   │   │   │   │   ├── prompt.svg
│   │   │   │   │   │   ├── proto.svg
│   │   │   │   │   │   ├── protractor.svg
│   │   │   │   │   │   ├── pug.svg
│   │   │   │   │   │   ├── puppeteer.svg
│   │   │   │   │   │   ├── puppet.svg
│   │   │   │   │   │   ├── purescript.svg
│   │   │   │   │   │   ├── python-misc.svg
│   │   │   │   │   │   ├── python.svg
│   │   │   │   │   │   ├── pytorch.svg
│   │   │   │   │   │   ├── qsharp.svg
│   │   │   │   │   │   ├── quarto.svg
│   │   │   │   │   │   ├── quasar.svg
│   │   │   │   │   │   ├── quokka.svg
│   │   │   │   │   │   ├── qwik.svg
│   │   │   │   │   │   ├── racket.svg
│   │   │   │   │   │   ├── raml.svg
│   │   │   │   │   │   ├── razor.svg
│   │   │   │   │   │   ├── rbxmk.svg
│   │   │   │   │   │   ├── rc.svg
│   │   │   │   │   │   ├── react.svg
│   │   │   │   │   │   ├── react_ts.svg
│   │   │   │   │   │   ├── readme.svg
│   │   │   │   │   │   ├── reason.svg
│   │   │   │   │   │   ├── red.svg
│   │   │   │   │   │   ├── redux-action.svg
│   │   │   │   │   │   ├── redux-reducer.svg
│   │   │   │   │   │   ├── redux-selector.svg
│   │   │   │   │   │   ├── redux-store.svg
│   │   │   │   │   │   ├── regedit.svg
│   │   │   │   │   │   ├── remark.svg
│   │   │   │   │   │   ├── remix_light.svg
│   │   │   │   │   │   ├── remix.svg
│   │   │   │   │   │   ├── renovate.svg
│   │   │   │   │   │   ├── replit.svg
│   │   │   │   │   │   ├── rescript-interface.svg
│   │   │   │   │   │   ├── rescript.svg
│   │   │   │   │   │   ├── restql.svg
│   │   │   │   │   │   ├── riot.svg
│   │   │   │   │   │   ├── roadmap.svg
│   │   │   │   │   │   ├── roblox.svg
│   │   │   │   │   │   ├── robots.svg
│   │   │   │   │   │   ├── robot.svg
│   │   │   │   │   │   ├── rocket.svg
│   │   │   │   │   │   ├── rojo.svg
│   │   │   │   │   │   ├── rollup.svg
│   │   │   │   │   │   ├── rome.svg
│   │   │   │   │   │   ├── routing.svg
│   │   │   │   │   │   ├── rspec.svg
│   │   │   │   │   │   ├── r.svg
│   │   │   │   │   │   ├── rubocop_light.svg
│   │   │   │   │   │   ├── rubocop.svg
│   │   │   │   │   │   ├── ruby.svg
│   │   │   │   │   │   ├── ruff.svg
│   │   │   │   │   │   ├── rust.svg
│   │   │   │   │   │   ├── salesforce.svg
│   │   │   │   │   │   ├── san.svg
│   │   │   │   │   │   ├── sass.svg
│   │   │   │   │   │   ├── sas.svg
│   │   │   │   │   │   ├── sbt.svg
│   │   │   │   │   │   ├── scala.svg
│   │   │   │   │   │   ├── scheme.svg
│   │   │   │   │   │   ├── scons_light.svg
│   │   │   │   │   │   ├── scons.svg
│   │   │   │   │   │   ├── screwdriver.svg
│   │   │   │   │   │   ├── search.svg
│   │   │   │   │   │   ├── semantic-release_light.svg
│   │   │   │   │   │   ├── semantic-release.svg
│   │   │   │   │   │   ├── semgrep.svg
│   │   │   │   │   │   ├── sentry.svg
│   │   │   │   │   │   ├── sequelize.svg
│   │   │   │   │   │   ├── serverless.svg
│   │   │   │   │   │   ├── settings.svg
│   │   │   │   │   │   ├── shader.svg
│   │   │   │   │   │   ├── silverstripe.svg
│   │   │   │   │   │   ├── simulink.svg
│   │   │   │   │   │   ├── siyuan.svg
│   │   │   │   │   │   ├── sketch.svg
│   │   │   │   │   │   ├── slim.svg
│   │   │   │   │   │   ├── slint.svg
│   │   │   │   │   │   ├── slug.svg
│   │   │   │   │   │   ├── smarty.svg
│   │   │   │   │   │   ├── sml.svg
│   │   │   │   │   │   ├── snakemake.svg
│   │   │   │   │   │   ├── snapcraft.svg
│   │   │   │   │   │   ├── snowpack_light.svg
│   │   │   │   │   │   ├── snowpack.svg
│   │   │   │   │   │   ├── snyk.svg
│   │   │   │   │   │   ├── solidity.svg
│   │   │   │   │   │   ├── sonarcloud.svg
│   │   │   │   │   │   ├── spwn.svg
│   │   │   │   │   │   ├── stackblitz.svg
│   │   │   │   │   │   ├── stan.svg
│   │   │   │   │   │   ├── steadybit.svg
│   │   │   │   │   │   ├── stencil.svg
│   │   │   │   │   │   ├── stitches_light.svg
│   │   │   │   │   │   ├── stitches.svg
│   │   │   │   │   │   ├── storybook.svg
│   │   │   │   │   │   ├── stryker.svg
│   │   │   │   │   │   ├── stylable.svg
│   │   │   │   │   │   ├── stylelint_light.svg
│   │   │   │   │   │   ├── stylelint.svg
│   │   │   │   │   │   ├── stylus.svg
│   │   │   │   │   │   ├── sublime.svg
│   │   │   │   │   │   ├── subtitles.svg
│   │   │   │   │   │   ├── supabase.svg
│   │   │   │   │   │   ├── svelte.svg
│   │   │   │   │   │   ├── svgo.svg
│   │   │   │   │   │   ├── svgr.svg
│   │   │   │   │   │   ├── svg.svg
│   │   │   │   │   │   ├── swagger.svg
│   │   │   │   │   │   ├── sway.svg
│   │   │   │   │   │   ├── swc.svg
│   │   │   │   │   │   ├── swift.svg
│   │   │   │   │   │   ├── syncpack.svg
│   │   │   │   │   │   ├── systemd_light.svg
│   │   │   │   │   │   ├── systemd.svg
│   │   │   │   │   │   ├── table.svg
│   │   │   │   │   │   ├── tailwindcss.svg
│   │   │   │   │   │   ├── taskfile.svg
│   │   │   │   │   │   ├── tauri.svg
│   │   │   │   │   │   ├── taze.svg
│   │   │   │   │   │   ├── tcl.svg
│   │   │   │   │   │   ├── teal.svg
│   │   │   │   │   │   ├── template.svg
│   │   │   │   │   │   ├── templ.svg
│   │   │   │   │   │   ├── terraform.svg
│   │   │   │   │   │   ├── test-js.svg
│   │   │   │   │   │   ├── test-jsx.svg
│   │   │   │   │   │   ├── test-ts.svg
│   │   │   │   │   │   ├── tex.svg
│   │   │   │   │   │   ├── textlint.svg
│   │   │   │   │   │   ├── tilt.svg
│   │   │   │   │   │   ├── tldraw_light.svg
│   │   │   │   │   │   ├── tldraw.svg
│   │   │   │   │   │   ├── tobimake.svg
│   │   │   │   │   │   ├── tobi.svg
│   │   │   │   │   │   ├── todo.svg
│   │   │   │   │   │   ├── toml_light.svg
│   │   │   │   │   │   ├── toml.svg
│   │   │   │   │   │   ├── travis.svg
│   │   │   │   │   │   ├── tree.svg
│   │   │   │   │   │   ├── trigger.svg
│   │   │   │   │   │   ├── tsconfig.svg
│   │   │   │   │   │   ├── tsdoc.svg
│   │   │   │   │   │   ├── tsil.svg
│   │   │   │   │   │   ├── tune.svg
│   │   │   │   │   │   ├── turborepo_light.svg
│   │   │   │   │   │   ├── turborepo.svg
│   │   │   │   │   │   ├── twig.svg
│   │   │   │   │   │   ├── twine.svg
│   │   │   │   │   │   ├── typescript-def.svg
│   │   │   │   │   │   ├── typescript.svg
│   │   │   │   │   │   ├── typst.svg
│   │   │   │   │   │   ├── umi.svg
│   │   │   │   │   │   ├── uml_light.svg
│   │   │   │   │   │   ├── uml.svg
│   │   │   │   │   │   ├── unity.svg
│   │   │   │   │   │   ├── unocss.svg
│   │   │   │   │   │   ├── url.svg
│   │   │   │   │   │   ├── uv.svg
│   │   │   │   │   │   ├── vagrant.svg
│   │   │   │   │   │   ├── vala.svg
│   │   │   │   │   │   ├── vanilla-extract.svg
│   │   │   │   │   │   ├── varnish.svg
│   │   │   │   │   │   ├── vedic.svg
│   │   │   │   │   │   ├── velite.svg
│   │   │   │   │   │   ├── velocity.svg
│   │   │   │   │   │   ├── vercel_light.svg
│   │   │   │   │   │   ├── vercel.svg
│   │   │   │   │   │   ├── verdaccio.svg
│   │   │   │   │   │   ├── verified.svg
│   │   │   │   │   │   ├── verilog.svg
│   │   │   │   │   │   ├── vfl.svg
│   │   │   │   │   │   ├── video.svg
│   │   │   │   │   │   ├── vim.svg
│   │   │   │   │   │   ├── virtual.svg
│   │   │   │   │   │   ├── visualstudio.svg
│   │   │   │   │   │   ├── vitest.svg
│   │   │   │   │   │   ├── vite.svg
│   │   │   │   │   │   ├── vlang.svg
│   │   │   │   │   │   ├── vscode.svg
│   │   │   │   │   │   ├── vue-config.svg
│   │   │   │   │   │   ├── vue.svg
│   │   │   │   │   │   ├── vuex-store.svg
│   │   │   │   │   │   ├── wakatime_light.svg
│   │   │   │   │   │   ├── wakatime.svg
│   │   │   │   │   │   ├── wallaby.svg
│   │   │   │   │   │   ├── wally.svg
│   │   │   │   │   │   ├── watchman.svg
│   │   │   │   │   │   ├── webassembly.svg
│   │   │   │   │   │   ├── webhint.svg
│   │   │   │   │   │   ├── webpack.svg
│   │   │   │   │   │   ├── wepy.svg
│   │   │   │   │   │   ├── werf.svg
│   │   │   │   │   │   ├── windicss.svg
│   │   │   │   │   │   ├── wolframlanguage.svg
│   │   │   │   │   │   ├── word.svg
│   │   │   │   │   │   ├── wrangler.svg
│   │   │   │   │   │   ├── wxt.svg
│   │   │   │   │   │   ├── xaml.svg
│   │   │   │   │   │   ├── xmake.svg
│   │   │   │   │   │   ├── xml.svg
│   │   │   │   │   │   ├── yaml.svg
│   │   │   │   │   │   ├── yang.svg
│   │   │   │   │   │   ├── yarn.svg
│   │   │   │   │   │   ├── zeabur_light.svg
│   │   │   │   │   │   ├── zeabur.svg
│   │   │   │   │   │   ├── zig.svg
│   │   │   │   │   │   └── zip.svg
│   │   │   │   │   └── provider
│   │   │   │   │       ├── abacus.svg
│   │   │   │   │       ├── aihubmix.svg
│   │   │   │   │       ├── alibaba-cn.svg
│   │   │   │   │       ├── alibaba.svg
│   │   │   │   │       ├── amazon-bedrock.svg
│   │   │   │   │       ├── anthropic.svg
│   │   │   │   │       ├── azure-cognitive-services.svg
│   │   │   │   │       ├── azure.svg
│   │   │   │   │       ├── bailing.svg
│   │   │   │   │       ├── baseten.svg
│   │   │   │   │       ├── cerebras.svg
│   │   │   │   │       ├── chutes.svg
│   │   │   │   │       ├── cloudflare-ai-gateway.svg
│   │   │   │   │       ├── cloudflare-workers-ai.svg
│   │   │   │   │       ├── cohere.svg
│   │   │   │   │       ├── cortecs.svg
│   │   │   │   │       ├── deepinfra.svg
│   │   │   │   │       ├── deepseek.svg
│   │   │   │   │       ├── fastrouter.svg
│   │   │   │   │       ├── fireworks-ai.svg
│   │   │   │   │       ├── friendli.svg
│   │   │   │   │       ├── github-copilot.svg
│   │   │   │   │       ├── github-models.svg
│   │   │   │   │       ├── google.svg
│   │   │   │   │       ├── google-vertex-anthropic.svg
│   │   │   │   │       ├── google-vertex.svg
│   │   │   │   │       ├── groq.svg
│   │   │   │   │       ├── helicone.svg
│   │   │   │   │       ├── huggingface.svg
│   │   │   │   │       ├── iflowcn.svg
│   │   │   │   │       ├── inception.svg
│   │   │   │   │       ├── inference.svg
│   │   │   │   │       ├── io-net.svg
│   │   │   │   │       ├── kimi-for-coding.svg
│   │   │   │   │       ├── llama.svg
│   │   │   │   │       ├── lmstudio.svg
│   │   │   │   │       ├── lucidquery.svg
│   │   │   │   │       ├── minimax-cn.svg
│   │   │   │   │       ├── minimax.svg
│   │   │   │   │       ├── mistral.svg
│   │   │   │   │       ├── modelscope.svg
│   │   │   │   │       ├── moonshotai-cn.svg
│   │   │   │   │       ├── moonshotai.svg
│   │   │   │   │       ├── morph.svg
│   │   │   │   │       ├── nano-gpt.svg
│   │   │   │   │       ├── nebius.svg
│   │   │   │   │       ├── nvidia.svg
│   │   │   │   │       ├── ollama-cloud.svg
│   │   │   │   │       ├── openai.svg
│   │   │   │   │       ├── opencode.svg
│   │   │   │   │       ├── openrouter.svg
│   │   │   │   │       ├── ovhcloud.svg
│   │   │   │   │       ├── perplexity.svg
│   │   │   │   │       ├── poe.svg
│   │   │   │   │       ├── requesty.svg
│   │   │   │   │       ├── sap-ai-core.svg
│   │   │   │   │       ├── scaleway.svg
│   │   │   │   │       ├── siliconflow-cn.svg
│   │   │   │   │       ├── siliconflow.svg
│   │   │   │   │       ├── submodel.svg
│   │   │   │   │       ├── synthetic.svg
│   │   │   │   │       ├── togetherai.svg
│   │   │   │   │       ├── upstage.svg
│   │   │   │   │       ├── v0.svg
│   │   │   │   │       ├── venice.svg
│   │   │   │   │       ├── vercel.svg
│   │   │   │   │       ├── vultr.svg
│   │   │   │   │       ├── wandb.svg
│   │   │   │   │       ├── xai.svg
│   │   │   │   │       ├── xiaomi.svg
│   │   │   │   │       ├── zai-coding-plan.svg
│   │   │   │   │       ├── zai.svg
│   │   │   │   │       ├── zenmux.svg
│   │   │   │   │       ├── zhipuai-coding-plan.svg
│   │   │   │   │       └── zhipuai.svg
│   │   │   │   └── images
│   │   │   │       ├── social-share-black.png
│   │   │   │       ├── social-share.png
│   │   │   │       └── social-share-zen.png
│   │   │   ├── components
│   │   │   │   ├── accordion.css
│   │   │   │   ├── accordion.tsx
│   │   │   │   ├── app-icon.css
│   │   │   │   ├── app-icons
│   │   │   │   │   ├── sprite.svg
│   │   │   │   │   └── types.ts
│   │   │   │   ├── app-icon.tsx
│   │   │   │   ├── avatar.css
│   │   │   │   ├── avatar.tsx
│   │   │   │   ├── basic-tool.css
│   │   │   │   ├── basic-tool.tsx
│   │   │   │   ├── button.css
│   │   │   │   ├── button.tsx
│   │   │   │   ├── card.css
│   │   │   │   ├── card.tsx
│   │   │   │   ├── checkbox.css
│   │   │   │   ├── checkbox.tsx
│   │   │   │   ├── code.css
│   │   │   │   ├── code.tsx
│   │   │   │   ├── collapsible.css
│   │   │   │   ├── collapsible.tsx
│   │   │   │   ├── context-menu.css
│   │   │   │   ├── context-menu.tsx
│   │   │   │   ├── dialog.css
│   │   │   │   ├── dialog.tsx
│   │   │   │   ├── diff-changes.css
│   │   │   │   ├── diff-changes.tsx
│   │   │   │   ├── diff.css
│   │   │   │   ├── diff-ssr.tsx
│   │   │   │   ├── diff.tsx
│   │   │   │   ├── dock-prompt.tsx
│   │   │   │   ├── dock-surface.css
│   │   │   │   ├── dock-surface.tsx
│   │   │   │   ├── dropdown-menu.css
│   │   │   │   ├── dropdown-menu.tsx
│   │   │   │   ├── favicon.tsx
│   │   │   │   ├── file-icon.css
│   │   │   │   ├── file-icons
│   │   │   │   │   ├── sprite.svg
│   │   │   │   │   └── types.ts
│   │   │   │   ├── file-icon.tsx
│   │   │   │   ├── font.tsx
│   │   │   │   ├── hover-card.css
│   │   │   │   ├── hover-card.tsx
│   │   │   │   ├── icon-button.css
│   │   │   │   ├── icon-button.tsx
│   │   │   │   ├── icon.css
│   │   │   │   ├── icon.tsx
│   │   │   │   ├── image-preview.css
│   │   │   │   ├── image-preview.tsx
│   │   │   │   ├── inline-input.css
│   │   │   │   ├── inline-input.tsx
│   │   │   │   ├── keybind.css
│   │   │   │   ├── keybind.tsx
│   │   │   │   ├── line-comment.css
│   │   │   │   ├── line-comment.tsx
│   │   │   │   ├── list.css
│   │   │   │   ├── list.tsx
│   │   │   │   ├── logo.css
│   │   │   │   ├── logo.tsx
│   │   │   │   ├── markdown.css
│   │   │   │   ├── markdown.tsx
│   │   │   │   ├── message-nav.css
│   │   │   │   ├── message-nav.tsx
│   │   │   │   ├── message-part.css
│   │   │   │   ├── message-part.tsx
│   │   │   │   ├── popover.css
│   │   │   │   ├── popover.tsx
│   │   │   │   ├── progress-circle.css
│   │   │   │   ├── progress-circle.tsx
│   │   │   │   ├── progress.css
│   │   │   │   ├── progress.tsx
│   │   │   │   ├── provider-icon.css
│   │   │   │   ├── provider-icons
│   │   │   │   │   ├── sprite.svg
│   │   │   │   │   └── types.ts
│   │   │   │   ├── provider-icon.tsx
│   │   │   │   ├── radio-group.css
│   │   │   │   ├── radio-group.tsx
│   │   │   │   ├── resize-handle.css
│   │   │   │   ├── resize-handle.tsx
│   │   │   │   ├── scroll-view.css
│   │   │   │   ├── scroll-view.tsx
│   │   │   │   ├── select.css
│   │   │   │   ├── select.tsx
│   │   │   │   ├── session-review.css
│   │   │   │   ├── session-review.tsx
│   │   │   │   ├── session-turn.css
│   │   │   │   ├── session-turn.tsx
│   │   │   │   ├── spinner.css
│   │   │   │   ├── spinner.tsx
│   │   │   │   ├── sticky-accordion-header.css
│   │   │   │   ├── sticky-accordion-header.tsx
│   │   │   │   ├── switch.css
│   │   │   │   ├── switch.tsx
│   │   │   │   ├── tabs.css
│   │   │   │   ├── tabs.tsx
│   │   │   │   ├── tag.css
│   │   │   │   ├── tag.tsx
│   │   │   │   ├── text-field.css
│   │   │   │   ├── text-field.tsx
│   │   │   │   ├── text-shimmer.css
│   │   │   │   ├── text-shimmer.tsx
│   │   │   │   ├── toast.css
│   │   │   │   ├── toast.tsx
│   │   │   │   ├── tooltip.css
│   │   │   │   ├── tooltip.tsx
│   │   │   │   ├── typewriter.css
│   │   │   │   └── typewriter.tsx
│   │   │   ├── context
│   │   │   │   ├── code.tsx
│   │   │   │   ├── data.tsx
│   │   │   │   ├── dialog.tsx
│   │   │   │   ├── diff.tsx
│   │   │   │   ├── helper.tsx
│   │   │   │   ├── i18n.tsx
│   │   │   │   ├── index.ts
│   │   │   │   ├── marked.tsx
│   │   │   │   └── worker-pool.tsx
│   │   │   ├── custom-elements.d.ts
│   │   │   ├── hooks
│   │   │   │   ├── create-auto-scroll.tsx
│   │   │   │   ├── index.ts
│   │   │   │   └── use-filtered-list.tsx
│   │   │   ├── i18n
│   │   │   │   ├── ar.ts
│   │   │   │   ├── br.ts
│   │   │   │   ├── bs.ts
│   │   │   │   ├── da.ts
│   │   │   │   ├── de.ts
│   │   │   │   ├── en.ts
│   │   │   │   ├── es.ts
│   │   │   │   ├── fr.ts
│   │   │   │   ├── ja.ts
│   │   │   │   ├── ko.ts
│   │   │   │   ├── no.ts
│   │   │   │   ├── pl.ts
│   │   │   │   ├── ru.ts
│   │   │   │   ├── th.ts
│   │   │   │   ├── zh.ts
│   │   │   │   └── zht.ts
│   │   │   ├── pierre
│   │   │   │   ├── index.ts
│   │   │   │   ├── virtualizer.ts
│   │   │   │   └── worker.ts
│   │   │   ├── styles
│   │   │   │   ├── animations.css
│   │   │   │   ├── base.css
│   │   │   │   ├── colors.css
│   │   │   │   ├── index.css
│   │   │   │   ├── tailwind
│   │   │   │   │   ├── colors.css
│   │   │   │   │   ├── index.css
│   │   │   │   │   └── utilities.css
│   │   │   │   ├── theme.css
│   │   │   │   └── utilities.css
│   │   │   └── theme
│   │   │       ├── color.ts
│   │   │       ├── context.tsx
│   │   │       ├── default-themes.ts
│   │   │       ├── desktop-theme.schema.json
│   │   │       ├── index.ts
│   │   │       ├── loader.ts
│   │   │       ├── resolve.ts
│   │   │       ├── themes
│   │   │       │   ├── aura.json
│   │   │       │   ├── ayu.json
│   │   │       │   ├── carbonfox.json
│   │   │       │   ├── catppuccin.json
│   │   │       │   ├── dracula.json
│   │   │       │   ├── gruvbox.json
│   │   │       │   ├── monokai.json
│   │   │       │   ├── nightowl.json
│   │   │       │   ├── nord.json
│   │   │       │   ├── oc-1.json
│   │   │       │   ├── oc-2.json
│   │   │       │   ├── onedarkpro.json
│   │   │       │   ├── shadesofpurple.json
│   │   │       │   ├── solarized.json
│   │   │       │   ├── tokyonight.json
│   │   │       │   └── vesper.json
│   │   │       └── types.ts
│   │   ├── sst-env.d.ts
│   │   ├── tsconfig.json
│   │   └── vite.config.ts
│   ├── util
│   │   ├── package.json
│   │   ├── src
│   │   │   ├── array.ts
│   │   │   ├── binary.ts
│   │   │   ├── encode.ts
│   │   │   ├── error.ts
│   │   │   ├── fn.ts
│   │   │   ├── identifier.ts
│   │   │   ├── iife.ts
│   │   │   ├── lazy.ts
│   │   │   ├── path.ts
│   │   │   ├── retry.ts
│   │   │   └── slug.ts
│   │   ├── sst-env.d.ts
│   │   └── tsconfig.json
│   └── web
│       ├── astro.config.mjs
│       ├── config.mjs
│       ├── package.json
│       ├── public
│       │   ├── apple-touch-icon.png -> ../../ui/src/assets/favicon/apple-touch-icon.png
│       │   ├── apple-touch-icon-v3.png -> ../../ui/src/assets/favicon/apple-touch-icon-v3.png
│       │   ├── favicon-96x96.png -> ../../ui/src/assets/favicon/favicon-96x96.png
│       │   ├── favicon-96x96-v3.png -> ../../ui/src/assets/favicon/favicon-96x96-v3.png
│       │   ├── favicon.ico -> ../../ui/src/assets/favicon/favicon.ico
│       │   ├── favicon.svg -> ../../ui/src/assets/favicon/favicon.svg
│       │   ├── favicon-v3.ico -> ../../ui/src/assets/favicon/favicon-v3.ico
│       │   ├── favicon-v3.svg -> ../../ui/src/assets/favicon/favicon-v3.svg
│       │   ├── robots.txt
│       │   ├── site.webmanifest -> ../../ui/src/assets/favicon/site.webmanifest
│       │   ├── social-share.png -> ../../ui/src/assets/images/social-share.png
│       │   ├── social-share-zen.png -> ../../ui/src/assets/images/social-share-zen.png
│       │   ├── theme.json
│       │   ├── web-app-manifest-192x192.png -> ../../ui/src/assets/favicon/web-app-manifest-192x192.png
│       │   └── web-app-manifest-512x512.png -> ../../ui/src/assets/favicon/web-app-manifest-512x512.png
│       ├── README.md
│       ├── src
│       │   ├── assets
│       │   │   ├── lander
│       │   │   │   ├── check.svg
│       │   │   │   ├── copy.svg
│       │   │   │   ├── screenshot-github.png
│       │   │   │   ├── screenshot.png
│       │   │   │   ├── screenshot-splash.png
│       │   │   │   └── screenshot-vscode.png
│       │   │   ├── logo-dark.svg
│       │   │   ├── logo-light.svg
│       │   │   ├── logo-ornate-dark.svg
│       │   │   ├── logo-ornate-light.svg
│       │   │   └── web
│       │   │       ├── web-homepage-active-session.png
│       │   │       ├── web-homepage-new-session.png
│       │   │       └── web-homepage-see-servers.png
│       │   ├── components
│       │   │   ├── Footer.astro
│       │   │   ├── Head.astro
│       │   │   ├── Header.astro
│       │   │   ├── Hero.astro
│       │   │   ├── icons
│       │   │   │   ├── custom.tsx
│       │   │   │   └── index.tsx
│       │   │   ├── Lander.astro
│       │   │   ├── share
│       │   │   │   ├── common.tsx
│       │   │   │   ├── content-bash.module.css
│       │   │   │   ├── content-bash.tsx
│       │   │   │   ├── content-code.module.css
│       │   │   │   ├── content-code.tsx
│       │   │   │   ├── content-diff.module.css
│       │   │   │   ├── content-diff.tsx
│       │   │   │   ├── content-error.module.css
│       │   │   │   ├── content-error.tsx
│       │   │   │   ├── content-markdown.module.css
│       │   │   │   ├── content-markdown.tsx
│       │   │   │   ├── content-text.module.css
│       │   │   │   ├── content-text.tsx
│       │   │   │   ├── copy-button.module.css
│       │   │   │   ├── copy-button.tsx
│       │   │   │   ├── part.module.css
│       │   │   │   └── part.tsx
│       │   │   ├── share.module.css
│       │   │   ├── Share.tsx
│       │   │   └── SiteTitle.astro
│       │   ├── content
│       │   │   ├── docs
│       │   │   │   ├── acp.mdx
│       │   │   │   ├── agents.mdx
│       │   │   │   ├── ar
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── bs
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── cli.mdx
│       │   │   │   ├── commands.mdx
│       │   │   │   ├── config.mdx
│       │   │   │   ├── custom-tools.mdx
│       │   │   │   ├── da
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── de
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── ecosystem.mdx
│       │   │   │   ├── enterprise.mdx
│       │   │   │   ├── es
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── formatters.mdx
│       │   │   │   ├── fr
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── github.mdx
│       │   │   │   ├── gitlab.mdx
│       │   │   │   ├── ide.mdx
│       │   │   │   ├── index.mdx
│       │   │   │   ├── it
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── ja
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── keybinds.mdx
│       │   │   │   ├── ko
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── lsp.mdx
│       │   │   │   ├── mcp-servers.mdx
│       │   │   │   ├── models.mdx
│       │   │   │   ├── modes.mdx
│       │   │   │   ├── nb
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── network.mdx
│       │   │   │   ├── permissions.mdx
│       │   │   │   ├── pl
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── plugins.mdx
│       │   │   │   ├── providers.mdx
│       │   │   │   ├── pt-br
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── ru
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── rules.mdx
│       │   │   │   ├── sdk.mdx
│       │   │   │   ├── server.mdx
│       │   │   │   ├── share.mdx
│       │   │   │   ├── skills.mdx
│       │   │   │   ├── th
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── themes.mdx
│       │   │   │   ├── tools.mdx
│       │   │   │   ├── tr
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   ├── troubleshooting.mdx
│       │   │   │   ├── tui.mdx
│       │   │   │   ├── web.mdx
│       │   │   │   ├── windows-wsl.mdx
│       │   │   │   ├── zen.mdx
│       │   │   │   ├── zh-cn
│       │   │   │   │   ├── acp.mdx
│       │   │   │   │   ├── agents.mdx
│       │   │   │   │   ├── cli.mdx
│       │   │   │   │   ├── commands.mdx
│       │   │   │   │   ├── config.mdx
│       │   │   │   │   ├── custom-tools.mdx
│       │   │   │   │   ├── ecosystem.mdx
│       │   │   │   │   ├── enterprise.mdx
│       │   │   │   │   ├── formatters.mdx
│       │   │   │   │   ├── github.mdx
│       │   │   │   │   ├── gitlab.mdx
│       │   │   │   │   ├── ide.mdx
│       │   │   │   │   ├── index.mdx
│       │   │   │   │   ├── keybinds.mdx
│       │   │   │   │   ├── lsp.mdx
│       │   │   │   │   ├── mcp-servers.mdx
│       │   │   │   │   ├── models.mdx
│       │   │   │   │   ├── modes.mdx
│       │   │   │   │   ├── network.mdx
│       │   │   │   │   ├── permissions.mdx
│       │   │   │   │   ├── plugins.mdx
│       │   │   │   │   ├── providers.mdx
│       │   │   │   │   ├── rules.mdx
│       │   │   │   │   ├── sdk.mdx
│       │   │   │   │   ├── server.mdx
│       │   │   │   │   ├── share.mdx
│       │   │   │   │   ├── skills.mdx
│       │   │   │   │   ├── themes.mdx
│       │   │   │   │   ├── tools.mdx
│       │   │   │   │   ├── troubleshooting.mdx
│       │   │   │   │   ├── tui.mdx
│       │   │   │   │   ├── web.mdx
│       │   │   │   │   ├── windows-wsl.mdx
│       │   │   │   │   └── zen.mdx
│       │   │   │   └── zh-tw
│       │   │   │       ├── acp.mdx
│       │   │   │       ├── agents.mdx
│       │   │   │       ├── cli.mdx
│       │   │   │       ├── commands.mdx
│       │   │   │       ├── config.mdx
│       │   │   │       ├── custom-tools.mdx
│       │   │   │       ├── ecosystem.mdx
│       │   │   │       ├── enterprise.mdx
│       │   │   │       ├── formatters.mdx
│       │   │   │       ├── github.mdx
│       │   │   │       ├── gitlab.mdx
│       │   │   │       ├── ide.mdx
│       │   │   │       ├── index.mdx
│       │   │   │       ├── keybinds.mdx
│       │   │   │       ├── lsp.mdx
│       │   │   │       ├── mcp-servers.mdx
│       │   │   │       ├── models.mdx
│       │   │   │       ├── modes.mdx
│       │   │   │       ├── network.mdx
│       │   │   │       ├── permissions.mdx
│       │   │   │       ├── plugins.mdx
│       │   │   │       ├── providers.mdx
│       │   │   │       ├── rules.mdx
│       │   │   │       ├── sdk.mdx
│       │   │   │       ├── server.mdx
│       │   │   │       ├── share.mdx
│       │   │   │       ├── skills.mdx
│       │   │   │       ├── themes.mdx
│       │   │   │       ├── tools.mdx
│       │   │   │       ├── troubleshooting.mdx
│       │   │   │       ├── tui.mdx
│       │   │   │       ├── web.mdx
│       │   │   │       ├── windows-wsl.mdx
│       │   │   │       └── zen.mdx
│       │   │   └── i18n
│       │   │       ├── ar.json
│       │   │       ├── bs.json
│       │   │       ├── da.json
│       │   │       ├── de.json
│       │   │       ├── en.json
│       │   │       ├── es.json
│       │   │       ├── fr.json
│       │   │       ├── it.json
│       │   │       ├── ja.json
│       │   │       ├── ko.json
│       │   │       ├── nb.json
│       │   │       ├── pl.json
│       │   │       ├── pt-BR.json
│       │   │       ├── ru.json
│       │   │       ├── th.json
│       │   │       ├── tr.json
│       │   │       ├── zh-CN.json
│       │   │       └── zh-TW.json
│       │   ├── content.config.ts
│       │   ├── i18n
│       │   │   └── locales.ts
│       │   ├── middleware.ts
│       │   ├── pages
│       │   │   ├── s
│       │   │   │   └── [id].astro
│       │   │   └── [...slug].md.ts
│       │   ├── styles
│       │   │   └── custom.css
│       │   └── types
│       │       ├── lang-map.d.ts
│       │       └── starlight-virtual.d.ts
│       ├── sst-env.d.ts
│       └── tsconfig.json
├── patches
│   ├── @openrouter%2Fai-sdk-provider@1.5.4.patch
│   └── @standard-community%2Fstandard-openapi@0.2.9.patch
├── README.ar.md
├── README.bn.md
├── README.br.md
├── README.bs.md
├── README.da.md
├── README.de.md
├── README.es.md
├── README.fr.md
├── README.it.md
├── README.ja.md
├── README.ko.md
├── README.md
├── README.no.md
├── README.pl.md
├── README.ru.md
├── README.th.md
├── README.tr.md
├── README.uk.md
├── README.zh.md
├── README.zht.md
├── script
│   ├── beta.ts
│   ├── changelog.ts
│   ├── duplicate-pr.ts
│   ├── format.ts
│   ├── generate.ts
│   ├── hooks
│   ├── publish.ts
│   ├── release
│   ├── stats.ts
│   ├── sync-zed.ts
│   └── version.ts
├── sdks
│   └── vscode
│       ├── bun.lock
│       ├── esbuild.js
│       ├── eslint.config.mjs
│       ├── images
│       │   ├── button-dark.svg -> ../../../packages/identity/mark.svg
│       │   ├── button-light.svg -> ../../../packages/identity/mark-light.svg
│       │   └── icon.png -> ../../../packages/identity/mark-512x512.png
│       ├── package.json
│       ├── README.md
│       ├── script
│       │   ├── publish
│       │   └── release
│       ├── src
│       │   └── extension.ts
│       ├── sst-env.d.ts
│       └── tsconfig.json
├── SECURITY.md
├── specs
│   ├── project.md
│   └── session-composer-refactor-plan.md
├── sst.config.ts
├── sst-env.d.ts
├── STATS.md
├── tsconfig.json
└── turbo.json

418 directories, 3849 files