# Admin Interface - Email 2FA Authentication

The admin interface is now secured with email-based two-factor authentication (2FA).

## Setup Instructions

### 1. Run the Setup Script

```bash
./scripts/setup_admin_credentials.sh
```

The script will prompt you for:
- **Username**: Your admin username
- **Password**: Your admin password (will be hidden)
- **Email**: Your email address for receiving 2FA codes

### 2. Verify Your Email in AWS SES

After running the setup script, you need to verify your email address:

```bash
aws ses verify-email-identity --email-address YOUR_EMAIL@example.com --region us-east-1
```

Check your email and click the verification link.

### 3. (Optional) Request SES Production Access

By default, AWS SES is in sandbox mode and can only send to verified addresses. To send to any email:

1. Go to https://console.aws.amazon.com/ses/
2. Navigate to "Account Dashboard"
3. Click "Request production access"
4. Fill out the form with your use case

## How It Works

### Login Flow

1. **Enter Credentials**: User enters username and password
2. **Generate 2FA Code**: System generates a 6-digit code
3. **Send Email**: Code is sent via AWS SES to your verified email
4. **Verify Code**: User enters the code within 10 minutes
5. **Create Session**: Valid code creates a 24-hour session token

### Security Features

- ✅ **No Fallback**: No default admin/admin credentials
- ✅ **Email 2FA**: Required for all logins when configured
- ✅ **Code Expiry**: 2FA codes expire after 10 minutes
- ✅ **One-Time Use**: Codes are deleted after successful verification
- ✅ **Session Management**: 24-hour sessions with automatic expiry
- ✅ **Secure Storage**: Credentials stored in AWS Secrets Manager

### Email Template

Users receive a professionally formatted email with:
- Large, easy-to-read 6-digit code
- 10-minute expiration warning
- Security notice about unsolicited codes

## Updating Credentials

To change your username, password, or email:

```bash
aws secretsmanager put-secret-value \
  --secret-id ai-video-codec/admin-credentials \
  --secret-string '{
    "username": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD",
    "email": "YOUR_EMAIL@example.com",
    "2fa_enabled": true
  }' \
  --region us-east-1
```

## Disabling 2FA (Not Recommended)

To disable 2FA and use only password authentication:

```bash
aws secretsmanager put-secret-value \
  --secret-id ai-video-codec/admin-credentials \
  --secret-string '{
    "username": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD",
    "2fa_enabled": false
  }' \
  --region us-east-1
```

## Troubleshooting

### Email Not Arriving

1. Check SES sending limits:
   ```bash
   aws ses get-send-quota --region us-east-1
   ```

2. Verify email is confirmed:
   ```bash
   aws ses get-identity-verification-attributes \
     --identities YOUR_EMAIL@example.com \
     --region us-east-1
   ```

3. Check CloudWatch Logs for the admin Lambda function

### Login Fails

1. Verify credentials are set:
   ```bash
   aws secretsmanager get-secret-value \
     --secret-id ai-video-codec/admin-credentials \
     --region us-east-1
   ```

2. Check Lambda logs:
   ```bash
   aws logs tail /aws/lambda/ai-video-codec-admin-api --follow
   ```

### 2FA Code Expired

Codes are valid for 10 minutes. If expired, restart the login process to receive a new code.

## Access

**URL**: https://aiv1codec.com/admin.html

After successful authentication, you'll have access to:
- System status monitoring
- Experiment control (start/stop)
- Autonomous mode management
- Real-time metrics updates

## Security Best Practices

1. **Use Strong Passwords**: Minimum 12 characters with mixed case, numbers, and symbols
2. **Keep Email Secure**: Use a secure email account with its own 2FA
3. **Regular Updates**: Change password periodically
4. **Monitor Access**: Check CloudWatch logs for suspicious activity
5. **Session Management**: Log out when done to invalidate tokens

