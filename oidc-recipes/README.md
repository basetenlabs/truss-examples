# OIDC Recipes

These examples use the OIDC token available to running model code through [runtime OIDC](https://docs.baseten.co/organization/oidc#use-oidc-at-request-time). They show how a model can access customer-owned resources without storing long-lived credentials.

The recipes use AWS, but the same pattern is available from other OIDC-compatible providers:

- [AWS](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_oidc.html)
- [GCP](https://docs.cloud.google.com/iam/docs/workload-identity-federation)
- [Azure](https://learn.microsoft.com/en-us/entra/workload-id/workload-identity-federation)
- [HashiCorp Vault](https://developer.hashicorp.com/vault/docs/auth/jwt#jwt-authentication)
- [Snowflake](https://docs.snowflake.com/en/user-guide/workload-identity-federation)
- [Databricks](https://docs.databricks.com/aws/en/dev-tools/auth/oauth-federation)

## Recipes

- [`oidc-fetch-a-resource`](oidc-fetch-a-resource): Fetch an object from S3 at request time.
- [`oidc-envelope-weight-encryption`](oidc-envelope-weight-encryption): Mount encrypted weights from S3 and decrypt them with AWS KMS during startup.

## Folder structure

Each recipe contains:

- `README.md`: What the recipe does and how to run it.
- `setup.sh`: Creates the example AWS resources and OIDC trust.
- `standard-truss/`: A regular Truss with `config.yaml` and `model/model.py`.
- `custom-base-image/`: An optional custom-server Truss. This is currently used by the envelope-weight-encryption recipe.

Fill in values marked `FILL ME` or left empty in `setup.sh` and `config.yaml`. Run the setup script first, copy its output into the Truss config, and then run `truss push` on the relevant Truss directory.
