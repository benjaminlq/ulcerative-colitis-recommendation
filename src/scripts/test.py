from importlib import import_module
import os

mod_path = os.path.join("prompts", "polyp", "polyp").replace("/",".")
mod = import_module(mod_path)

print(mod.PROMPT_TEMPLATE)