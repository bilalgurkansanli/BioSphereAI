/**
 * The open-source project, in one place.
 *
 * Every link on the public site that points at the repository, its documents or its licence
 * is built from these constants, so moving the repository is a one-line change here rather
 * than a hunt through the copy.
 */

export const REPO_URL = "https://github.com/bilalgurkansanli/BioSphereAI";

/** Where the project page itself is published. */
export const SITE_URL = "https://biosphere.bilalgurkansanli.com";

/**
 * Built as the public project page only (`NEXT_PUBLIC_SITE_ONLY=1`): no engine stands behind
 * it, so the dashboard, capture and embed routes redirect to the guide (see next.config.ts)
 * and the site stops linking to them. Unset, the same code is the full local app.
 */
export const SITE_ONLY = process.env.NEXT_PUBLIC_SITE_ONLY === "1";

/** `owner/name`, for the places that print the repository rather than link to it. */
export const REPO_SLUG = REPO_URL.replace(/^https:\/\/github\.com\//, "");

/** The directory `git clone` creates. */
export const REPO_NAME = REPO_SLUG.split("/").pop() ?? "BioSphereAI";

/** The branch the documentation links resolve against. */
export const DEFAULT_BRANCH = "main";

export const LICENSE = "MIT";

export const AUTHOR = {
  handle: "bilalgurkansanli",
  name: "Bilal Gürkan Şanlı",
  url: "https://github.com/bilalgurkansanli",
} as const;

const blob = (path: string) => `${REPO_URL}/blob/${DEFAULT_BRANCH}/${path}`;

export const DOCS = {
  readme: `${REPO_URL}#readme`,
  readmeTr: blob("README.tr.md"),
  architecture: blob("docs/ARCHITECTURE.md"),
  contributing: blob("CONTRIBUTING.md"),
  license: blob("LICENSE"),
  issues: `${REPO_URL}/issues`,
  newIssue: `${REPO_URL}/issues/new`,
  helpWanted: `${REPO_URL}/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22help%20wanted%22`,
  pulls: `${REPO_URL}/pulls`,
  setupScript: blob("scripts/setup.ps1"),
  envExample: blob("engine/.env.example"),
} as const;

/** What `git clone` takes. */
export const CLONE_URL = `${REPO_URL}.git`;
