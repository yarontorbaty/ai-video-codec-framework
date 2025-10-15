# AI Video Codec Framework - Live Dashboard

A real-time monitoring dashboard for the AI Video Codec Framework, displaying progress, usage, and cost metrics.

## 🎯 Features

### Real-Time Metrics
- **Experiments**: Total count, recent activity, success rate
- **Compression**: Best compression ratio achieved vs target (90%)
- **Quality**: Best PSNR achieved vs target (95%)
- **Costs**: Monthly spending vs budget ($5,000)

### Infrastructure Monitoring
- **Orchestrator**: Master controller status and performance
- **Training Workers**: Auto-scaling GPU instances for model training
- **Inference Workers**: Real-time processing instances
- **Storage**: S3 buckets, EFS filesystem, DynamoDB tables

### Visual Analytics
- **Cost Breakdown**: Pie chart showing spending by component
- **Progress Charts**: Compression and quality improvement over time
- **Experiment Table**: Recent experiments with status and results

### Responsive Design
- **Mobile Friendly**: Optimized for phones and tablets
- **Desktop**: Full-featured dashboard for large screens
- **Dark Mode**: Automatic dark mode support

## 🚀 Quick Start

### Deploy Dashboard
```bash
# Deploy to AWS S3 + CloudFront
./scripts/deploy_dashboard.sh
```

### Access Dashboard
The deployment script will output the dashboard URL:
```
Dashboard URL: https://d1234567890.cloudfront.net
```

## 📊 Dashboard Sections

### Overview Cards
- **Total Experiments**: Number of experiments run
- **Best Compression**: Highest bitrate reduction achieved
- **Best Quality**: Highest PSNR score achieved
- **Monthly Cost**: Current month spending

### Infrastructure Status
- **Orchestrator**: CPU, memory, status
- **Training Workers**: Active instances, queue depth, spot usage
- **Inference Workers**: Active instances, queue depth, FPS
- **Storage**: S3 objects, EFS usage, DynamoDB items

### Recent Experiments
Table showing:
- Experiment ID
- Status (running, completed, failed)
- Compression ratio achieved
- Quality (PSNR) score
- Duration
- Cost

### Cost Breakdown
- **Training Workers**: ~60% of costs
- **Inference Workers**: ~20% of costs
- **Storage**: ~10% of costs
- **Orchestrator**: ~10% of costs

### Performance Charts
- **Compression Progress**: Line chart showing improvement over time
- **Quality Progress**: Line chart showing PSNR improvement

## 🔧 Technical Details

### Architecture
```
Internet → CloudFront → S3 Bucket (Static Files)
                    ↓
              API Gateway → Lambda → AWS Services
```

### Data Sources
- **CloudFormation**: Infrastructure status
- **DynamoDB**: Experiment metadata
- **SQS**: Queue depths and messages
- **Cost Explorer**: Spending data
- **CloudWatch**: Performance metrics

### Auto-Refresh
- **Interval**: 30 seconds
- **Real-time**: Connection status indicator
- **Error Handling**: Graceful degradation

## 🎨 Customization

### Styling
Edit `dashboard/styles.css` to customize:
- Colors and themes
- Layout and spacing
- Responsive breakpoints
- Dark mode support

### Functionality
Edit `dashboard/app.js` to add:
- New metrics
- Custom charts
- Additional data sources
- Real-time updates

### Content
Edit `dashboard/index.html` to modify:
- Dashboard sections
- Layout structure
- Meta information

## 📱 Mobile Support

The dashboard is fully responsive and works on:
- **Phones**: Optimized layout for small screens
- **Tablets**: Balanced layout for medium screens
- **Desktops**: Full-featured dashboard

## 🔒 Security

### Public Access
- **Static Files**: Publicly accessible via CloudFront
- **API Endpoints**: CORS enabled for cross-origin requests
- **No Authentication**: Public dashboard for transparency

### Data Privacy
- **No Sensitive Data**: Only metrics and progress information
- **No Credentials**: No AWS keys or secrets exposed
- **Aggregated Data**: Only summary statistics displayed

## 🚨 Troubleshooting

### Dashboard Not Loading
```bash
# Check CloudFront distribution
aws cloudfront get-distribution --id DISTRIBUTION_ID

# Check S3 bucket
aws s3 ls s3://BUCKET_NAME/

# Check API Gateway
aws apigateway get-rest-api --rest-api-id API_ID
```

### No Data Displayed
```bash
# Check Lambda function logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/ai-video-codec-dashboard-api"

# Check API Gateway logs
aws logs describe-log-groups --log-group-name-prefix "/aws/apigateway"
```

### Performance Issues
```bash
# Check CloudFront cache
aws cloudfront get-distribution --id DISTRIBUTION_ID --query 'Distribution.DistributionConfig.CacheBehaviors'

# Invalidate cache
aws cloudfront create-invalidation --distribution-id DISTRIBUTION_ID --paths "/*"
```

## 📈 Monitoring

### CloudWatch Metrics
- **Lambda Duration**: API response times
- **CloudFront Requests**: Dashboard access
- **S3 Requests**: Static file access
- **API Gateway**: API usage

### Cost Tracking
- **CloudFront**: ~$1-5/month for traffic
- **S3**: ~$0.50/month for storage
- **Lambda**: ~$0.10/month for API calls
- **API Gateway**: ~$0.50/month for requests

## 🔄 Updates

### Deploy Changes
```bash
# Update dashboard files
./scripts/deploy_dashboard.sh
```

### Version Control
- Dashboard files are version controlled
- Changes tracked in git
- Rollback available via git

## 📚 Documentation

### Related Files
- `infrastructure/cloudformation/dashboard.yaml` - CloudFormation template
- `scripts/deploy_dashboard.sh` - Deployment script
- `dashboard/` - Static files directory

### External Links
- [CloudFront Documentation](https://docs.aws.amazon.com/cloudfront/)
- [S3 Static Website Hosting](https://docs.aws.amazon.com/s3/latest/userguide/WebsiteHosting.html)
- [API Gateway Documentation](https://docs.aws.amazon.com/apigateway/)

## 🎉 Success Metrics

### Dashboard Performance
- **Load Time**: < 2 seconds
- **Uptime**: 99.9% availability
- **Mobile Score**: 90+ Lighthouse score
- **Accessibility**: WCAG 2.1 AA compliant

### User Experience
- **Real-time Updates**: 30-second refresh
- **Responsive Design**: Works on all devices
- **Error Handling**: Graceful degradation
- **Visual Appeal**: Modern, professional design

---

**Dashboard Status**: ✅ Ready for deployment  
**Last Updated**: October 15, 2025  
**Version**: 1.0.0
