# CLAUDE.md

Alle projektweiten Regeln, das Projektziel und der aktuelle Stand stehen in **AGENTS.md**. Sie gelten für Claude Code genauso wie für jeden anderen Agenten:

@AGENTS.md

## Ergänzungen für Claude Code

- Shell-Befehle laufen unter Windows (Git Bash oder PowerShell). Für Python-Ausgaben mit Sonderzeichen `PYTHONIOENCODING=utf-8` voranstellen.
- Lange mehrzeilige Python-Patches nicht als Bash-Heredoc mit verschachtelten Anführungszeichen schreiben. Besser ein Skript im Scratchpad anlegen oder das Edit-Tool nutzen.
- Nach `gh pr create` den PR-Status über die PR-Tools der App lesen, nicht CI per Polling abfragen.
