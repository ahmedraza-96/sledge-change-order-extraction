# Deploying to EC2 with Docker and GitHub Actions

Every push to `main` runs `.github/workflows/deploy.yml`:

1. **test**: the offline test suite (no API keys needed).
2. **deploy**:
   - GitHub gets short-lived AWS credentials through an IAM role (OIDC). No AWS keys are stored in GitHub.
   - It builds the Docker image and pushes it to ECR.
   - It tells the EC2 instance, through SSM Run Command, to pull the image and restart the
     `co-extract` container. The instance's SSH port doesn't need to be open to GitHub.

API keys live only on the instance, in `/opt/co-extract/.env`. They're never in the image or in GitHub.

## Current deployment

- Instance `kestrel-recon` (`i-0c8315ea4cff28197`, ap-south-1). It also runs the kestrel-recon app on port 8501.
- ECR repository `co-extract`, deploy role `co-extract-github-deploy`.
- The app is at `http://52.66.247.74:8000` (that public IP changes if the instance is stopped and started).
- The run history (SQLite) is in `/opt/co-extract/data/history.db` on the instance. The deploy mounts that
  folder into the container, so redeploys keep it. To copy it to your machine for review:
  `scp <user>@<instance-ip>:/opt/co-extract/data/history.db .` (or use `sqlite3` on the instance). To clear it,
  delete the file and restart the container (`docker restart co-extract`).

These values are defaults in `deploy.yml`. Set the repository variables only to deploy somewhere else.

## One-time setup

**1. AWS resources.** You need a working `aws` login, then:

```bash
AWS_REGION=<your region> INSTANCE_NAME=fyp-deployment ./deploy/aws-setup.sh
```

This creates or updates the ECR repository, the GitHub OIDC provider and deploy role (trusted only by
this repo's `main` branch), and gives the instance's role SSM and ECR-read access. If the instance
already has a role, the script adds to it rather than replacing it.

**2. The instance** needs Docker, the AWS CLI and the SSM agent. Amazon Linux 2023 and recent Ubuntu
AMIs ship the SSM agent. On Amazon Linux 2023:

```bash
sudo dnf install -y docker && sudo systemctl enable --now docker
sudo mkdir -p /opt/co-extract && sudo nano /opt/co-extract/.env   # OPENROUTER_API_KEY=... (same format as .env.example)
sudo chmod 600 /opt/co-extract/.env
```

**3. Network.** Open the app port (default **8000**; set the `APP_PORT` variable to change it) in the
instance's security group. Better still, put it behind the reverse proxy the instance already uses.

**4. GitHub repository variables.** The script prints the exact `gh variable set` commands:
`AWS_REGION`, `AWS_ROLE_ARN`, `ECR_REPOSITORY`, `EC2_INSTANCE_ID`, and optionally `APP_PORT`.

## Deploy

Push to `main`, or run the workflow by hand: **Actions → deploy → Run workflow**. Check it worked:

```bash
curl http://<instance-public-ip>:8000/api/samples
```
