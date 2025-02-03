import axios, { AxiosError } from 'axios';
import * as dotenv from 'dotenv';
import * as fs from 'fs/promises';
import * as path from 'path';
import FormData from 'form-data';

dotenv.config();

async function searchForStockUrl(query: string): Promise<string | null> {
  try {
    const searxngUrl = process.env.SEARXNG_URL;
    if (!searxngUrl) {
      throw new Error('SEARXNG_URL not configured in environment');
    }

    console.log('Searching for:', query);
    const response = await axios.get(`${searxngUrl}/search`, {
      params: {
        q: query,
        format: 'json',
        engines: 'google',
        language: 'en',
        time_range: 'day'
      }
    });

    if (response.data && response.data.results && response.data.results.length > 0) {
      // Get the first result URL
      const url = response.data.results[0].url;
      console.log('Found URL:', url);
      return url;
    }

    console.log('No results found');
    return null;
  } catch (error) {
    console.error('Error searching with SearxNG:', error instanceof Error ? error.message : error);
    return null;
  }
}

async function testScraper() {
  try {
    console.log('Starting scraper test...');

    // Use a simple URL first to test the vision model
    const url = 'https://finance.yahoo.com/quote/AAPL';
    
    console.log('Testing scraper with URL:', url);
    const response = await axios.post('http://localhost:3000/scrape', {
      url
    }, {
      timeout: 60000 // Increase timeout to 60 seconds
    });
    
    console.log('Response:', JSON.stringify(response.data, null, 2));
  } catch (error) {
    if (error instanceof AxiosError) {
      console.error('Error testing scraper:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        headers: error.response?.headers,
        config: {
          url: error.config?.url,
          method: error.config?.method,
          data: error.config?.data
        }
      });
    } else {
      console.error('Unknown error:', error);
    }
  }
}

async function testVisionModel() {
    try {
        // Get the base URL without /v1
        const ollamaUrl = (process.env.OLLAMA_URL || '').replace(/\/v1\/?$/, '').replace(/\/$/, '');
        if (!ollamaUrl) {
            throw new Error('OLLAMA_URL not configured');
        }

        // Read test image
        const imagePath = path.join(__dirname, 'screenshots', 'screenshot_0.png');
        const imageBuffer = await fs.readFile(imagePath);

        // Create request body following the blog example
        const requestBody = {
            model: process.env.OLLAMA_VISION_MODEL || 'llama3.2-vision:11b-instruct-q8_0',
            messages: [{
                role: 'user',
                content: 'What financial information can you see in this image?',
                images: [imageBuffer.toString('base64')]
            }],
            stream: true // Enable streaming
        };

        console.log('Sending request to Ollama vision model...');
        const response = await axios.post(`${ollamaUrl}/api/chat`, requestBody, {
            timeout: 120000,
            maxBodyLength: Infinity,
            maxContentLength: Infinity,
            responseType: 'stream'
        });

        let fullContent = '';
        
        // Handle streaming response
        response.data.on('data', (chunk: Buffer) => {
            const lines = chunk.toString().split('\n').filter(line => line.trim());
            for (const line of lines) {
                try {
                    const data = JSON.parse(line);
                    if (data.message?.content) {
                        process.stdout.write(data.message.content);
                        fullContent += data.message.content;
                    }
                } catch (e) {
                    // Ignore parse errors for incomplete chunks
                }
            }
        });

        // Wait for stream to end
        await new Promise((resolve, reject) => {
            response.data.on('end', () => {
                console.log('\n\nFull response content:', fullContent);
                resolve(fullContent);
            });
            response.data.on('error', reject);
        });

    } catch (error) {
        console.error('Error testing vision model:', error instanceof Error ? error.message : error);
    }
}

testScraper();
testVisionModel(); 