import express from 'express';
import type { Request, Response, RequestHandler } from 'express';
import { WebScraper } from './puppeteer';

const app = express();
const port = 3000;

// Initialize scraper
const scraper = new WebScraper();

// Middleware to parse JSON bodies
app.use(express.json());

// Add logging middleware
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} ${req.method} ${req.url}`);
    next();
});

// Health check endpoint
app.get('/health', (_req: Request, res: Response) => {
    res.json({ status: 'ok' });
});

interface ScrapeRequest {
    url: string;
}

// Scraping endpoint
const scrapeHandler: RequestHandler = async (req, res) => {
    const { url } = req.body as ScrapeRequest;
    
    if (!url) {
        console.log('Error: URL is required');
        res.status(400).json({ error: 'URL is required' });
        return;
    }
    
    try {
        console.log(`Starting to scrape ${url}...`);
        const content = await scraper.scrapeUrl(url);
        console.log('Content retrieved successfully:', {
            contentLength: content?.length || 0,
            excerpt: content?.substring(0, 100)
        });
        res.json({ content });
    } catch (error) {
        const errorDetails = error instanceof Error ? {
            message: error.message,
            stack: error.stack,
            name: error.name
        } : error;
        console.error('Error scraping URL:', errorDetails);
        res.status(500).json({ error: 'Failed to scrape URL', details: errorDetails });
    }
};

app.post('/scrape', scrapeHandler);

// Initialize the scraper and start the server
async function main() {
    try {
        // Initialize the scraper
        await scraper.init();
        console.log('Scraper initialized successfully');
        
        // Start the server
        app.listen(port, () => {
            console.log(`Scraper server running at http://localhost:${port}`);
        });
        
    } catch (error) {
        console.error('Failed to initialize scraper:', error);
        process.exit(1);
    }
}

// Handle shutdown gracefully
process.on('SIGINT', async () => {
    console.log('Shutting down gracefully...');
    await scraper.close();
    process.exit(0);
});

// Run the server
main(); 