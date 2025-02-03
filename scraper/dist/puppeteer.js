"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.WebScraper = void 0;
const puppeteer = __importStar(require("puppeteer"));
const vision_scraper_1 = require("./vision-scraper");
class WebScraper {
    constructor(options = {}) {
        this.browser = null;
        this.visionScraper = null;
        this.options = {
            launchOptions: {
                headless: true,
                args: ['--no-sandbox', '--disable-setuid-sandbox'],
                ...(options.launchOptions || {})
            },
            gotoOptions: {
                waitUntil: 'networkidle0',
                timeout: 30000,
                ...(options.gotoOptions || {})
            },
            useVisionFallback: true,
            ...options
        };
    }
    async init() {
        this.browser = await puppeteer.launch(this.options.launchOptions);
        if (this.options.useVisionFallback) {
            this.visionScraper = new vision_scraper_1.VisionScraper();
        }
    }
    async scrapeUrl(url) {
        try {
            await this.init();
            const page = await this.browser.newPage();
            // Set a reasonable timeout
            await page.setDefaultNavigationTimeout(30000);
            // Navigate to the URL
            await page.goto(url, this.options.gotoOptions);
            // Extract relevant content based on common financial website selectors
            const content = await page.evaluate(() => {
                // Helper function to clean text
                const cleanText = (text) => {
                    return text
                        .replace(/\s+/g, ' ')
                        .replace(/\n+/g, ' ')
                        .trim();
                };
                // Array to store relevant content
                let relevantContent = [];
                // Common selectors for stock-related content
                const selectors = {
                    // Article content
                    articleSelectors: [
                        'article',
                        '[role="article"]',
                        '.article-content',
                        '.article__content',
                        '.article-body',
                        '.story-content'
                    ],
                    // Stock specific content
                    stockSelectors: [
                        '.stock-price',
                        '.quote-data',
                        '.market-data',
                        '.stock-summary',
                        '[data-test="quote-header"]',
                        '[data-test="quote-summary"]'
                    ],
                    // News content
                    newsSelectors: [
                        '.news-content',
                        '.news-story',
                        '.news-article',
                        '.story-body'
                    ],
                    // Analysis content
                    analysisSelectors: [
                        '.analysis',
                        '.market-analysis',
                        '.stock-analysis',
                        '.research-content'
                    ]
                };
                // Function to extract text from elements matching a selector
                const extractFromSelectors = (selectorList) => {
                    selectorList.forEach(selector => {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => {
                            const text = cleanText(el.textContent || '');
                            if (text && text.length > 20) { // Ignore very short snippets
                                relevantContent.push(text);
                            }
                        });
                    });
                };
                // Extract content using all selector groups
                Object.values(selectors).forEach(selectorGroup => {
                    extractFromSelectors(selectorGroup);
                });
                // If no content found with specific selectors, try to get main content
                if (relevantContent.length === 0) {
                    const mainContent = document.querySelector('main') || document.querySelector('.main-content');
                    if (mainContent) {
                        relevantContent.push(cleanText(mainContent.textContent || ''));
                    }
                }
                // Filter out duplicate content and join with newlines
                return Array.from(new Set(relevantContent))
                    .filter(text => {
                    // Keep only text that likely contains stock-related information
                    const stockKeywords = [
                        'stock', 'market', 'price', 'share', 'trading',
                        'investor', 'earnings', 'revenue', 'growth',
                        'analyst', 'forecast', 'dividend', 'company',
                        'sector', 'industry', 'performance'
                    ];
                    return stockKeywords.some(keyword => text.toLowerCase().includes(keyword));
                })
                    .join('\n\n');
            });
            await page.close();
            // If no content was extracted, try vision scraping
            if (!content && this.options.useVisionFallback && this.visionScraper) {
                console.log('No content extracted with Puppeteer, falling back to vision scraping...');
                return await this.visionScraper.scrapeUrl(url);
            }
            return content;
        }
        catch (error) {
            console.error(`Error scraping ${url} with Puppeteer:`, error);
            // If vision fallback is enabled, try vision scraping
            if (this.options.useVisionFallback && this.visionScraper) {
                console.log('Error with Puppeteer scraping, falling back to vision scraping...');
                return await this.visionScraper.scrapeUrl(url);
            }
            throw error;
        }
    }
    async takeScreenshot(url) {
        try {
            await this.init();
            const page = await this.browser.newPage();
            await page.goto(url, this.options.gotoOptions);
            await page.waitForSelector('body');
            const screenshot = await page.screenshot({ encoding: 'base64' });
            await page.close();
            return screenshot.toString();
        }
        catch (error) {
            console.error(`Error taking screenshot of ${url}:`, error);
            throw error;
        }
    }
    async close() {
        if (this.browser) {
            await this.browser.close();
            this.browser = null;
        }
        if (this.visionScraper) {
            await this.visionScraper.close();
            this.visionScraper = null;
        }
    }
}
exports.WebScraper = WebScraper;
