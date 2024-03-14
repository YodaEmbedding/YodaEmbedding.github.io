"""
Convert:

.. code-block:: html

    <div id="ref-balle2018variational" class="csl-entry">
    <span class="csl-left-margin">\[1\] </span><span class="csl-right-inline">J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a scale hyperprior,” in *Proc. ICLR*, 2018. Available: <https://arxiv.org/abs/1802.01436></span>
    </div>

To:

.. code-block:: markdown

    [^ref-balle2018variational]: J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a scale hyperprior,” in *Proc. ICLR*, 2018. Available: <https://arxiv.org/abs/1802.01436>

"""

import re


def parse(lines):
    curr_entry_id = None
    curr_entry_contents = None

    for line in lines:
        m = re.match(r'<div id="(ref-.*?)" class="csl-entry">', line)
        if m:
            curr_entry_id = m.group(1)
            continue

        pattern = (
            r'<span class="csl-left-margin">\\\[.*?\\\] </span>'
            r'<span class="csl-right-inline">(.*?)</span>'
        )
        m = re.match(pattern, line)
        if m:
            curr_entry_contents = m.group(1)
            continue

        m = re.match(r"</div>", line)
        if m:
            if curr_entry_id is None or curr_entry_contents is None:
                continue
            yield f"[^{curr_entry_id}]: {curr_entry_contents}"
            yield ""
            curr_entry_id = None
            curr_entry_contents = None
            continue


def main():
    with open("references.md") as f:
        lines = f.readlines()

    with open("references.md", "w") as f:
        for line in parse(lines):
            print(line, file=f)


if __name__ == "__main__":
    main()
