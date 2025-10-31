# Fire App Deployment Guide

This guide covers deploying your Flask app to various cloud platforms for mobile frontend integration.

## Prerequisites

1. **Environment Variables**: Set up your API keys
2. **Model Files**: Ensure all index files are included in deployment
3. **Dependencies**: All requirements are in `requirements.txt`

## Environment Variables

Create a `.env` file or set these in your hosting platform:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDINGS_BACKEND=openai
EMBEDDINGS_MODEL=text-embedding-3-small
```

## Deployment Options

### Option 1: Render (Recommended for Startups)

**Pros**: Free tier, easy setup, automatic HTTPS
**Cons**: Limited resources on free tier

1. **Sign up** at [render.com](https://render.com)
2. **Connect your GitHub repository**
3. **Create a new Web Service**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn wsgi:app`
   - Environment: Python 3.9+
4. **Set environment variables** in Render dashboard
5. **Deploy** - your app will be available at `https://your-app-name.onrender.com`

### Option 2: Railway

**Pros**: Simple deployment, good free tier
**Cons**: Limited bandwidth on free tier

1. **Sign up** at [railway.app](https://railway.app)
2. **Connect GitHub repository**
3. **Deploy automatically** - Railway detects Flask apps
4. **Set environment variables** in Railway dashboard
5. **Get your URL**: `https://your-app-name.railway.app`

### Option 3: Fly.io

**Pros**: Global deployment, generous free tier
**Cons**: More complex setup

1. **Install Fly CLI**: `curl -L https://fly.io/install.sh | sh`
2. **Login**: `fly auth login`
3. **Create app**: `fly launch`
4. **Deploy**: `fly deploy`
5. **Get URL**: `https://your-app-name.fly.dev`

### Option 4: AWS Lightsail (Production)

**Pros**: Full control, scalable, reliable
**Cons**: More complex, costs money

1. **Create Lightsail instance** (Ubuntu 20.04)
2. **SSH into instance** and install dependencies
3. **Clone repository** and set up environment
4. **Use PM2 or systemd** to run the app
5. **Set up domain** and SSL certificate

### Option 5: Google Cloud Run

**Pros**: Serverless, auto-scaling, pay-per-use
**Cons**: Cold starts, more complex setup

1. **Install Google Cloud CLI**
2. **Build and push Docker image**
3. **Deploy to Cloud Run**
4. **Set environment variables**

## Mobile App Integration

### API Endpoints

Your mobile app can call these endpoints:

```javascript
// Base URL: https://your-app-url.com

// Ask a question
POST /ask
{
  "question": "Where is the first responder loop?",
  "model": "Tesla_Model_3"
}

// Get images
GET /images/{filename}

// Get documents
GET /docs/{filename}

// Health check
GET /health
```

### Example Mobile App Code (React Native)

```javascript
const API_BASE_URL = 'https://your-app-url.com';

const askQuestion = async (question, model) => {
  try {
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: question,
        model: model
      })
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
};

// Usage
const result = await askQuestion(
  "Where is the first responder loop?", 
  "Tesla_Model_3"
);
console.log(result.answer);
console.log(result.images);
```

### Example Mobile App Code (Flutter)

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

class FireAppAPI {
  static const String baseUrl = 'https://your-app-url.com';
  
  static Future<Map<String, dynamic>> askQuestion(
    String question, 
    String model
  ) async {
    final response = await http.post(
      Uri.parse('$baseUrl/ask'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'question': question,
        'model': model,
      }),
    );
    
    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to get answer');
    }
  }
}

// Usage
final result = await FireAppAPI.askQuestion(
  "Where is the first responder loop?", 
  "Tesla_Model_3"
);
print(result['answer']);
print(result['images']);
```

## Security Considerations

1. **Rate Limiting**: Implement rate limiting for production
2. **API Keys**: Never expose API keys in client code
3. **CORS**: Configure CORS properly for your domain
4. **HTTPS**: Always use HTTPS in production
5. **Input Validation**: Validate all inputs on server side

## Monitoring

1. **Health Checks**: Use `/health` endpoint for monitoring
2. **Logs**: Monitor application logs for errors
3. **Performance**: Track response times and resource usage
4. **Uptime**: Set up uptime monitoring

## Scaling Considerations

1. **Caching**: Implement Redis for session/response caching
2. **Load Balancing**: Use multiple instances for high traffic
3. **CDN**: Use CDN for static assets (images, documents)
4. **Database**: Consider moving to a proper database for user data

## Cost Estimation

- **Render**: $7/month for basic plan
- **Railway**: $5/month for basic plan  
- **Fly.io**: Free tier (3 apps, 3GB RAM)
- **AWS Lightsail**: $3.50/month for basic instance
- **Google Cloud Run**: Pay-per-use (very cheap for low traffic)

## Next Steps

1. Choose a hosting platform
2. Set up environment variables
3. Deploy your app
4. Test the API endpoints
5. Integrate with your mobile app
6. Monitor and optimize performance

