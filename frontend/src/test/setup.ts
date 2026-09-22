import { afterEach } from "vitest";

import "@testing-library/jest-dom/vitest";
import { resetWorkspaceDrafts } from "../state/drafts";

afterEach(() => {
  resetWorkspaceDrafts();
});
