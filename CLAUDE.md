# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal Jekyll blog ("My TU Brain") deployed via GitHub Pages at https://thvinhtruong.github.io. There is no application code — only Jekyll content, layouts, and configuration. GitHub Pages builds and deploys automatically on push to the default branch; no CI workflow lives in this repo.

## Common commands

```bash
# Local preview (Jekyll + github-pages gem from Gemfile)
bundle install
bundle exec jekyll serve            # http://localhost:4000, live reload on file save
bundle exec jekyll build            # one-shot build into _site/

# New post (handles slug, date, and frontmatter)
bin/new-post "Post title"                       # no tags/categories
bin/new-post "Title" -c dev -t go,cli           # comma-separated cats + tags
```

`bin/new-post` writes to `_posts/YYYY-MM-DD-<slug>.md` and refuses to overwrite an existing file. Prefer it over hand-creating posts so the frontmatter stays consistent.

## Authoring conventions

- Posts live in `_posts/` and must be named `YYYY-MM-DD-title.md`. The permalink format is `/:year/:month/:day/:title/` (see `_config.yml`).
- Posts get `layout: post` automatically via the `defaults` block in `_config.yml` — don't add it manually unless overriding.
- Required frontmatter: `title`, `date` (with timezone offset, e.g. `+0700`), `categories`, `tags`. Categories and tags are YAML lists.
- Top-level pages (`about.md`, `categories.html`, `tags.html`) appear in the site header via the `header_pages` list in `_config.yml`. Add new top-level pages there to surface them in nav.

## Layout structure

- `_layouts/default.html` is the outer shell; `home.html`, `page.html`, `post.html` extend it.
- `_includes/` holds `head.html`, `header.html`, `footer.html` — shared across layouts.
- `assets/css/` is the only styling. There's no JS build step, no bundler, no Node toolchain.

## Plugins

Only the three GitHub-Pages-supported plugins are enabled: `jekyll-feed`, `jekyll-seo-tag`, `jekyll-sitemap`. Adding plugins outside the github-pages allowlist will build locally but break the deployed site.

## Publishing

There is a `/blog-it` skill that ingests a markdown file into `_posts/` and commits + pushes. Use it when the user asks to publish a note from elsewhere on disk. Otherwise, a manual `git push` is the deploy.
