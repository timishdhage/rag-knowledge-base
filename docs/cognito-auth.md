# Cognito authentication

The live API accepts `Authorization: Bearer <Cognito JWT>`.

Configure these values in the deployment secret store; do not commit them:

- `AWS_REGION=eu-central-1`
- `COGNITO_USER_POOL_ID`
- `COGNITO_APP_CLIENT_ID`
- Optional `COGNITO_ISSUER`; otherwise it is derived from the region and pool ID.

The API fetches and caches the user pool JWKS, verifies the RS256 signature, validates issuer and expiry, checks token use and client ID, and uses the verified `sub` claim as the document owner. Clients must not provide or override an owner ID.
