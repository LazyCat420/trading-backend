"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const puppeteer_1 = require("./puppeteer");
const app = (0, express_1.default)();
const port = 3000;
// Initialize scraper
const scraper = new puppeteer_1.WebScraper();
// Middleware to parse JSON bodies
app.use(express_1.default.json());
// Health check endpoint
app.get('/health', (_req, res) => {
    res.json({ status: 'ok' });
});
// Scraping endpoint
const scrapeHandler = async (req, res) => {
    const { url } = req.body;
    if (!url) {
        res.status(400).json({ error: 'URL is required' });
        return;
    }
    try {
        console.log(`Scraping ${url}...`);
        const content = await scraper.scrapeUrl(url);
        console.log('Content retrieved successfully');
        res.json({ content });
    }
    catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'Failed to scrape URL' });
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
    }
    catch (error) {
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
