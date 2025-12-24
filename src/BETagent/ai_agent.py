from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Iterable, Union
import os


def _default_llm_stub(prompt: str) -> str:
    """Default stub used when no LLM invoker is provided.

    It simply echoes the prompt and notes that a real LLM should be
    provided via the `llm_invoke` argument.
    """
    return (
        "[LLM STUB] No LLM client provided.\n"
        "Provide an `llm_invoke(prompt: str) -> str` callable to call a real LLM.\n\n"
        "Prompt received:\n" + prompt
    )


def analyze_bet_outputs(
  folder: Optional[Union[str, Path]] = None,
  *,
  files: Optional[Iterable[Union[str, Path]]] = None,
  prompt: str = "",
  llm_invoke: Optional[Callable[[str], str]] = None,
  output_basename: Optional[str] = None,
) -> str:
    """Read BET output files from `folder`, send them to an LLM, print and save response.

    Args:
      folder: Path to the folder containing the BET output files (the folder
        created by the BET process, e.g. `bet_analysis_outputs/...`).
      prompt: The pre-prompt to send to the LLM. Leave blank to fill in later.
      llm_invoke: A callable that accepts a single `prompt` string and returns
        the LLM's text response. If `None`, a local stub will be used.
      output_basename: Optional basename for the saved analysis file. If not
        provided, a timestamped name will be used.

    Returns:
      The LLM response string.

    The function will:
      - gather all files in `folder` (reads .json, .txt and other files),
      - create a combined prompt by appending file contents to the supplied
        `prompt` (which may be empty),
      - call `llm_invoke(combined_prompt)` to get analysis,
      - print the LLM response, and
      - save the response as a text file in `folder`.
    """
    # Determine source files: `files` argument takes precedence.
    source_paths: list[Path] = []

    if files:
      for f in files:
        p = Path(f)
        if not p.is_absolute():
          p = Path.cwd() / p
        if p.exists() and p.is_file():
          source_paths.append(p)
        else:
          raise FileNotFoundError(f"File not found: {p}")
    else:
      # If a folder was provided and is a file, read that single file.
      if folder:
        base = Path(folder)
        if base.exists() and base.is_file():
          source_paths = [base]
        elif base.exists() and base.is_dir():
          source_paths = sorted([p for p in base.iterdir() if p.is_file()])
        else:
          raise FileNotFoundError(f"Folder not found: {base}")
      else:
        # Auto-detect default bet outputs folder in cwd
        default = Path.cwd() / "bet_analysis_outputs"
        if not default.exists() or not default.is_dir():
          raise FileNotFoundError(
            "No `files` provided and default folder 'bet_analysis_outputs' not found in CWD"
          )
        source_paths = sorted([p for p in default.iterdir() if p.is_file()])

    # Read all source files
    parts: list[str] = []
    for p in source_paths:
      try:
        if p.suffix.lower() == ".json":
          data = json.loads(p.read_text(encoding="utf-8"))
          parts.append(f"--- {p.name} (json) ---\n{json.dumps(data, indent=2)}\n")
        else:
          text = p.read_text(encoding="utf-8", errors="replace")
          parts.append(f"--- {p.name} ---\n{text}\n")
      except Exception as exc:
        parts.append(f"--- {p.name} (failed to read: {exc}) ---\n")

    files_combined = "\n".join(parts)

    # Build the final prompt to send to the LLM. The user may leave `prompt`
    # empty and paste a pre-prompt later when calling this function.
    combined_prompt = prompt + "\n\n" + "[BEGIN ATTACHED FILES]\n" + files_combined + "\n[END ATTACHED FILES]"

    # Use provided LLM invoker or the stub
    invoker = llm_invoke or _default_llm_stub

    response = invoker(combined_prompt)

    # Print to stdout
    print(response)

    # Save response to a timestamped text file in the same folder as inputs
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = output_basename or f"BEtagent_research_{ts}.txt"
    out_dir = source_paths[0].parent if source_paths else Path.cwd()
    out_path = out_dir / name
    out_path.write_text(response, encoding="utf-8")

    return response


__all__ = ["analyze_bet_outputs"]


def make_gemini_invoker(api_key: Optional[str] = None, model: str = "gemini-3-flash-preview") -> Callable[[str], str]:
  """Return a callable that invokes Google Gemini using `google.genai`.

  This function tries to use `google.genai` and falls back to older
  `google.generativeai` if present. Set `GOOGLE_API_KEY` or pass `api_key`.

  Install the client with `pip install google-genai` (or the package
  name recommended by Google at the time you install).
  """
  genai = None
  legacy = None
  try:
    import google.genai as genai  # type: ignore
  except Exception:
    try:
      import google.generativeai as legacy  # type: ignore
    except Exception as e:  # pragma: no cover - runtime dependency
      raise ImportError(
        "google.genai (or google.generativeai) is required for Gemini invoker."
      ) from e

  key = api_key or os.getenv("GOOGLE_API_KEY")
  if not key:
    raise ValueError("No API key provided. Set GOOGLE_API_KEY or pass api_key")

  # Configure whichever SDK is available
  if genai is not None:
    # Newer SDKs often provide a `Client` or `configure` method; try common patterns
    try:
      if hasattr(genai, "configure"):
        genai.configure(api_key=key)
    except Exception:
      # ignore configure errors; some SDKs use client objects
      pass
  else:
    # legacy package
    try:
      if hasattr(legacy, "configure"):
        legacy.configure(api_key=key)
    except Exception:
      pass

  def invoker(prompt: str) -> str:
    try:
      if genai is not None:
        # google.genai: try client-based interface first
        try:
          client = genai.Client(api_key=key) if hasattr(genai, "Client") else genai
          if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            # Use models.generate_content (newer google.genai)
            resp = client.models.generate_content(
              model=model,
              contents=prompt
            )
          elif hasattr(client, "generate_content"):
            resp = client.generate_content(prompt)
          elif hasattr(client, "generate_text"):
            resp = client.generate_text(model=model, prompt=prompt)
          elif hasattr(client, "generate"):
            resp = client.generate(model=model, prompt=prompt)
          else:
            raise RuntimeError("No generate method found")
        except Exception:
          # Fallback to direct module functions
          if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=model, prompt=prompt)
          elif hasattr(genai, "generate"):
            resp = genai.generate(model=model, prompt=prompt)
          else:
            raise RuntimeError("Unsupported google.genai interface")
      else:
        # legacy google.generativeai
        if hasattr(legacy, "generate_text"):
          resp = legacy.generate_text(model=model, prompt=prompt)
        elif hasattr(legacy, "generate"):
          resp = legacy.generate(model=model, prompt=prompt)
        else:
          raise RuntimeError("Unsupported legacy google.generativeai interface")
    except Exception as exc:
      raise RuntimeError(f"Gemini request failed: {exc}") from exc

    # Extract text from response—handle multiple SDK response shapes
    text_content = None

    # Try direct text attributes
    for attr in ["text", "output", "output_text", "content"]:
      if hasattr(resp, attr):
        val = getattr(resp, attr)
        if isinstance(val, str) and val.strip():
          text_content = val
          break

    # Try nested structures (candidates, parts, etc.)
    if not text_content:
      try:
        if hasattr(resp, "candidates") and resp.candidates:
          cand = resp.candidates[0] if isinstance(resp.candidates, list) else resp.candidates
          if hasattr(cand, "content"):
            content = cand.content
            if hasattr(content, "parts") and content.parts:
              part = content.parts[0]
              if hasattr(part, "text"):
                text_content = part.text
              elif isinstance(part, str):
                text_content = part
            elif isinstance(content, str):
              text_content = content
          elif hasattr(cand, "text"):
            text_content = cand.text
      except Exception:
        pass

    # Try dict-like access
    if not text_content and isinstance(resp, dict):
      try:
        if "candidates" in resp and resp["candidates"]:
          cand = resp["candidates"][0]
          if isinstance(cand, dict):
            text_content = cand.get("content") or cand.get("text")
          elif hasattr(cand, "text"):
            text_content = cand.text
        elif "text" in resp:
          text_content = resp["text"]
        elif "output" in resp:
          text_content = resp["output"]
      except Exception:
        pass

    # If still no text, avoid object repr
    if not text_content:
      resp_str = str(resp)
      if "<" in resp_str and "object at" in resp_str:
        text_content = "[Error: Unable to extract text from LLM response. Got object repr instead.]"
      else:
        text_content = resp_str

    return text_content or "[Error: Empty response from LLM]"

  return invoker
