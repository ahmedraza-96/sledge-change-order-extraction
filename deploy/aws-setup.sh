#!/usr/bin/env bash
# One-time AWS setup for the GitHub Actions deploy (.github/workflows/deploy.yml).
# Idempotent: safe to re-run. Needs a working `aws` login with IAM, ECR and EC2 rights.
#
#   AWS_REGION=us-east-1 INSTANCE_NAME=fyp-deployment ./deploy/aws-setup.sh
#
# Creates or updates:
#   - ECR repository            co-extract
#   - GitHub OIDC provider      token.actions.githubusercontent.com (if the account lacks one)
#   - IAM role for GitHub       co-extract-github-deploy, trusted only by this repo's main branch,
#                               allowed to push to that ECR repo and send SSM commands to the instance
#   - Instance permissions      SSM agent access + ECR read on the instance's existing role
#                               (a new role/profile only if the instance has none)
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION, e.g. us-east-1}"
INSTANCE_NAME="${INSTANCE_NAME:-fyp-deployment}"
REPO="${GITHUB_REPO:-ahmedraza-96/sledge-change-order-extraction}"
ECR_REPOSITORY="${ECR_REPOSITORY:-co-extract}"
ROLE_NAME="${ROLE_NAME:-co-extract-github-deploy}"
export AWS_DEFAULT_REGION="$AWS_REGION"

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].InstanceId" --output text)
[ "$INSTANCE_ID" != "None" ] || { echo "No running instance named $INSTANCE_NAME in $AWS_REGION"; exit 1; }
echo "account $ACCOUNT · instance $INSTANCE_ID ($INSTANCE_NAME) · region $AWS_REGION"

# --- ECR repository (scan on push, keep the last 10 images) ---
aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$ECR_REPOSITORY" --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr put-lifecycle-policy --repository-name "$ECR_REPOSITORY" --lifecycle-policy-text \
  '{"rules":[{"rulePriority":1,"description":"keep last 10","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}}]}' >/dev/null
echo "ECR repository: $ECR_REPOSITORY"

# --- GitHub OIDC provider ---
OIDC_ARN="arn:aws:iam::$ACCOUNT:oidc-provider/token.actions.githubusercontent.com"
aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1 || \
  aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com \
    --client-id-list sts.amazonaws.com >/dev/null
echo "OIDC provider: $OIDC_ARN"

# --- Role GitHub Actions assumes: only this repo's main branch ---
TRUST=$(cat <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Federated":"$OIDC_ARN"},
 "Action":"sts:AssumeRoleWithWebIdentity","Condition":{
  "StringEquals":{"token.actions.githubusercontent.com:aud":"sts.amazonaws.com",
                  "token.actions.githubusercontent.com:sub":"repo:$REPO:ref:refs/heads/main"}}}]}
EOF
)
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name "$ROLE_NAME" --policy-document "$TRUST"
else
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST" \
    --description "GitHub Actions deploy for $REPO" >/dev/null
fi
PERMS=$(cat <<EOF
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":"ecr:GetAuthorizationToken","Resource":"*"},
 {"Effect":"Allow","Action":["ecr:BatchCheckLayerAvailability","ecr:InitiateLayerUpload","ecr:UploadLayerPart",
   "ecr:CompleteLayerUpload","ecr:PutImage","ecr:BatchGetImage","ecr:GetDownloadUrlForLayer"],
  "Resource":"arn:aws:ecr:$AWS_REGION:$ACCOUNT:repository/$ECR_REPOSITORY"},
 {"Effect":"Allow","Action":"ssm:SendCommand","Resource":[
   "arn:aws:ec2:$AWS_REGION:$ACCOUNT:instance/$INSTANCE_ID",
   "arn:aws:ssm:$AWS_REGION::document/AWS-RunShellScript"]},
 {"Effect":"Allow","Action":["ssm:GetCommandInvocation","ssm:ListCommandInvocations"],"Resource":"*"}]}
EOF
)
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name deploy --policy-document "$PERMS"
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)
echo "GitHub role: $ROLE_ARN"

# --- The instance needs SSM (to receive commands) and ECR read (to pull the image) ---
PROFILE_ARN=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].IamInstanceProfile.Arn" --output text)
if [ "$PROFILE_ARN" = "None" ]; then
  INSTANCE_ROLE=co-extract-ec2
  aws iam get-role --role-name "$INSTANCE_ROLE" >/dev/null 2>&1 || aws iam create-role --role-name "$INSTANCE_ROLE" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam get-instance-profile --instance-profile-name "$INSTANCE_ROLE" >/dev/null 2>&1 || {
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_ROLE" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$INSTANCE_ROLE" --role-name "$INSTANCE_ROLE"
    sleep 10  # let IAM propagate before attaching
  }
  aws ec2 associate-iam-instance-profile --instance-id "$INSTANCE_ID" --iam-instance-profile Name="$INSTANCE_ROLE" >/dev/null
else
  # The instance already has a role (e.g. for the FYP app): add to it rather than replace it.
  INSTANCE_ROLE=$(aws iam get-instance-profile --instance-profile-name "${PROFILE_ARN##*/}" \
    --query "InstanceProfile.Roles[0].RoleName" --output text)
fi
aws iam attach-role-policy --role-name "$INSTANCE_ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam attach-role-policy --role-name "$INSTANCE_ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
echo "instance role: $INSTANCE_ROLE (SSM + ECR read attached)"

cat <<EOF

Done. Set these as GitHub repository variables (or let Claude run the gh commands):
  gh variable set AWS_REGION       --body "$AWS_REGION"
  gh variable set AWS_ROLE_ARN     --body "$ROLE_ARN"
  gh variable set ECR_REPOSITORY   --body "$ECR_REPOSITORY"
  gh variable set EC2_INSTANCE_ID  --body "$INSTANCE_ID"
EOF
