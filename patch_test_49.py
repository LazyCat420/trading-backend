import re

with open("/home/lazycat/.gemini/antigravity-ide/brain/8688c264-ae05-49b9-bbb2-ba31ef193916/task.md", "r") as f:
    content = f.read()

content = content.replace("- `[/]` Prompt & AI / ML Regression", "- `[x]` Prompt & AI / ML Regression")
content = content.replace("- `[ ]` Cross-Repo Contract & Integration", "- `[x]` Cross-Repo Contract & Integration")
content = content.replace("- `[ ]` Mutation & Logic Coverage", "- `[x]` Mutation & Logic Coverage")

with open("/home/lazycat/.gemini/antigravity-ide/brain/8688c264-ae05-49b9-bbb2-ba31ef193916/task.md", "w") as f:
    f.write(content)
