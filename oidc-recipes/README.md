# OIDC Recipes
The examples in this folder are recipes for using the OIDC token available to your running model code ([runtime OIDC](https://docs.baseten.co/organization/oidc#use-oidc-at-request-time)).  

These examples will all use the AWS ecosystem unless otherwise specified. However, much of the functionality is available similarly from other providers. See the following list of popular OIDC-supporting providers and their documentation on setting up OIDC integration:
- [AWS](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_oidc.html) 
- [GCP](https://docs.cloud.google.com/iam/docs/workload-identity-federation) 
- [Azure](https://learn.microsoft.com/en-us/entra/workload-id/workload-identity-federation) 
- [HashiCorp Vault](https://developer.hashicorp.com/vault/docs/auth/jwt#jwt-authentication) ("JWT Authentication") 
- [Snowflake](https://docs.snowflake.com/en/user-guide/workload-identity-federation) 
- [Databricks](https://docs.databricks.com/aws/en/dev-tools/auth/oauth-federation) 

Genearlly, these are recipes - expect more one-shot, quickstart-esque scripts for you to copy/paste and get started seeing the results. 
For a receipe/subfolder here, expect the following structure:
- `README.md` file: expect an overview of the use case and any setup steps/scripts needed. Also, if there are particular details on implementing the recipe for a particular use case, those will be detailed here.
- `python` folder: expect traditional `model/model.py` truss deployments alongside a `config.yaml`, so you can run `truss push` / `baseten model push` directly.
- `bash` folder: expect `shell` scripts which can work almost universally in any custom `base_image` or as a startup hook from popular base images like `vLLM`.
Files called `fill_me` are configuration details you need to specify for the scripts to work.

As an auth primitve conforming to the OIDC spec, the Baseten OIDC token is incredibly flexible in enabling rich and secure integrations to outside systems. See the subfolders here to get started!