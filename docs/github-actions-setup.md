# GitHub Actions Setup for Daily Hurricane Report

## Required Repository Secrets

To use the automated daily hurricane report workflow, you need to configure the following secrets in your GitHub repository:

### Setting up Secrets
1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** for each of the following:

### Email Configuration Secrets

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `EMAIL_HOST` | SMTP server hostname | `smtp.gmail.com` |
| `EMAIL_PORT` | SMTP server port | `465` |
| `EMAIL_USERNAME` | SMTP login username | `your-email@gmail.com` |
| `EMAIL_PASSWORD` | SMTP login password | `your-app-password` |
| `EMAIL_FROM` | Sender email address | `data.science@humdata.org` |
| `EMAIL_TO` | Recipient email address(es) | `zachary.arno@un.org` |

### Important Notes

#### For Gmail Users:
- Use an **App Password** instead of your regular password
- Enable 2-factor authentication first
- Generate an app password: Google Account → Security → 2-Step Verification → App passwords

#### For Other Email Providers:
- **Outlook/Hotmail**: Use `smtp-mail.outlook.com` port `587`
- **Yahoo**: Use `smtp.mail.yahoo.com` port `587` or `465`
- **Custom SMTP**: Contact your IT team for correct settings

## Workflow Schedule

The workflow runs:
- **Daily at 6:00 AM UTC** (adjust the cron schedule in the YAML if needed)
- **Manually** via GitHub Actions tab → "Run workflow"
- **Automatically** when you push changes to Report.qmd or the email script

## Monitoring

- Check the **Actions** tab in your GitHub repository to see workflow runs
- Failed runs will show detailed error logs
- The generated report is always saved as an artifact (available for 30 days)
- Email sending failures won't stop the report generation

## Testing

To test the workflow:
1. Set up all the required secrets
2. Go to **Actions** → **Daily Hurricane Report** → **Run workflow**
3. Check the logs to ensure everything works correctly

## Customization

You can modify the workflow by editing `.github/workflows/daily-hurricane-report.yml`:
- Change the schedule time (modify the cron expression)
- Add more recipients
- Adjust the Python version
- Add additional dependencies
