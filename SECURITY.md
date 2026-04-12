# Security Policy

## Secrets

Never commit:

- OpenAI API keys
- Neo4j passwords
- `.env` files
- local database snapshots containing credentials

Use environment variables or the application UI for local secret injection.

## Responsible Handling of Generated Artifacts

Generated artifacts may contain:

- user-uploaded filenames
- runtime prompts
- model responses
- execution outputs

Treat benchmark artifacts and runtime outputs as potentially sensitive. Do not publish them casually.

## Reporting

If you discover a security-sensitive issue in this repository or deployment workflow, report it privately to the maintainers instead of opening a public issue with exploit details.

