# Security and privacy

This repository intentionally excludes all personal source material and derived event data.

Never commit raw writing, biographies, notes, chat exports, relationship graphs, credentials, private indexes, replay cases, predictions, scores, or experiment reports. The corresponding local directories are ignored by Git, but contributors must still inspect staged changes before every push.

Cloud baselines are opt-in. Keep provider credentials in environment variables and review every compiled case before using `--allow-cloud-upload`.

If personal or secret material is committed, do not merely delete it in a later commit. Treat the Git history as compromised, rotate exposed credentials, and rebuild or purge the repository history before changing repository visibility.
