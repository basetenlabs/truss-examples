#!/usr/bin/env bash
set -euo pipefail

# FILL ME
AWS_ROLE_ARN="arn:aws:iam::863478709086:role/B10OIDCRole" # e.g. arn:aws:iam::123456789012:role/BasetenOIDCRole
AWS_REGION="us-west-2"                                    # e.g. us-west-2
S3_BUCKET="863478709086-foo"                              # e.g. mybucket
S3_KEY="bar"                                              # e.g. inputs/review.txt

require_command() {
	if ! command -v "$1" >/dev/null 2>&1; then
		echo "error: required command '$1' was not found" >&2
		exit 1
	fi
}

require_value() {
	if [[ -z "$2" ]]; then
		echo "error: $1 must be set" >&2
		exit 1
	fi
}

xml_value() {
	local xml="$1"
	local tag="$2"
	sed -n "s:.*<${tag}>\([^<]*\)</${tag}>.*:\1:p" <<<"${xml}"
}

uri_encode_path() {
	local value="$1"
	local encoded=""
	local char
	local hex
	local index
	local LC_ALL=C

	for ((index = 0; index < ${#value}; index++)); do
		char="${value:index:1}"
		case "${char}" in
			[a-zA-Z0-9.~_/-]) encoded+="${char}" ;;
			*)
				printf -v hex '%02X' "'${char}"
				encoded+="%${hex}"
				;;
		esac
	done

	printf '%s' "${encoded}"
}

assume_role_with_web_identity() {
	local oidc_token="$1"
	local response

	response=$(curl --silent --show-error --fail \
		--request POST \
		--data-urlencode "Action=AssumeRoleWithWebIdentity" \
		--data-urlencode "Version=2011-06-15" \
		--data-urlencode "RoleArn=${AWS_ROLE_ARN}" \
		--data-urlencode "RoleSessionName=baseten-oidc-recipe" \
		--data-urlencode "WebIdentityToken=${oidc_token}" \
		"https://sts.${AWS_REGION}.amazonaws.com/") || {
			echo "error: STS rejected the OIDC token exchange" >&2
			exit 1
		}

	AWS_ACCESS_KEY_ID=$(xml_value "${response}" "AccessKeyId")
	AWS_SECRET_ACCESS_KEY=$(xml_value "${response}" "SecretAccessKey")
	AWS_SESSION_TOKEN=$(xml_value "${response}" "SessionToken")

	if [[ -z "${AWS_ACCESS_KEY_ID}" || -z "${AWS_SECRET_ACCESS_KEY}" || -z "${AWS_SESSION_TOKEN}" ]]; then
		echo "error: STS response did not contain temporary credentials" >&2
		exit 1
	fi
}

get_s3_object() {
	local host="s3.${AWS_REGION}.amazonaws.com"
	local object_path

	object_path=$(uri_encode_path "/${S3_BUCKET}/${S3_KEY}")

	curl --silent --show-error --fail \
		--aws-sigv4 "aws:amz:${AWS_REGION}:s3" \
		--user "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}" \
		--header "x-amz-security-token: ${AWS_SESSION_TOKEN}" \
		--path-as-is \
		"https://${host}${object_path}"
}

main() {
	local oidc_token_file="${B10_OIDC_TOKEN_PATH:-}"
	local oidc_token
	local text

	require_command curl
	require_command jq
	require_command sed
	require_value AWS_ROLE_ARN "${AWS_ROLE_ARN}"
	require_value AWS_REGION "${AWS_REGION}"
	require_value S3_BUCKET "${S3_BUCKET}"
	require_value S3_KEY "${S3_KEY}"
	require_value B10_OIDC_TOKEN_PATH "${oidc_token_file}"

	if [[ ! -r "${oidc_token_file}" ]]; then
		echo "error: OIDC token file is not readable: ${oidc_token_file}" >&2
		exit 1
	fi

	oidc_token=$(<"${oidc_token_file}")
	require_value "OIDC token" "${oidc_token}"

	assume_role_with_web_identity "${oidc_token}"
	text=$(get_s3_object)
	jq --null-input --compact-output --arg text "${text}" '{text: $text}'
}

main "$@"
