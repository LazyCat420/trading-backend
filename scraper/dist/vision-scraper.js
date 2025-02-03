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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.VisionScraper = void 0;
const selenium_webdriver_1 = require("selenium-webdriver");
const chrome_1 = require("selenium-webdriver/chrome");
const fs = __importStar(require("fs/promises"));
const path = __importStar(require("path"));
const axios_1 = __importDefault(require("axios"));
const form_data_1 = __importDefault(require("form-data"));
const dotenv = __importStar(require("dotenv"));
dotenv.config();
class VisionScraper {
    constructor(options = {}) {
        this.driver = null;
        this.options = {
            screenshotDir: path.join(process.cwd(), 'screenshots'),
            scrollDelay: 1000,
            maxScrolls: 5,
            ...options
        };
    }
    async init() {
        const chromeOptions = new chrome_1.Options();
        chromeOptions.addArguments('--headless=new', '--no-sandbox', '--disable-dev-shm-usage');
        this.driver = await new selenium_webdriver_1.Builder()
            .forBrowser('chrome')
            .setChromeOptions(chromeOptions)
            .build();
        // Ensure screenshot directory exists
        await fs.mkdir(this.options.screenshotDir, { recursive: true });
    }
    async takeFullPageScreenshot() {
        if (!this.driver)
            throw new Error('Driver not initialized');
        const screenshots = [];
        let lastHeight = 0;
        let scrollCount = 0;
        while (scrollCount < this.options.maxScrolls) {
            // Take screenshot of current viewport
            const screenshot = await this.driver.takeScreenshot();
            screenshots.push(screenshot);
            // Scroll down
            const currentHeight = await this.driver.executeScript('return document.body.scrollHeight');
            await this.driver.executeScript('window.scrollTo(0, document.body.scrollHeight)');
            // Wait for scroll
            await new Promise(resolve => setTimeout(resolve, this.options.scrollDelay));
            // Check if we've reached the bottom
            if (currentHeight === lastHeight)
                break;
            lastHeight = currentHeight;
            scrollCount++;
        }
        return screenshots;
    }
    async analyzeWithVisionModel(screenshots) {
        const ollamaUrl = process.env.OLLAMA_URL;
        const visionModel = process.env.OLLAMA_VISION_MODEL;
        if (!ollamaUrl || !visionModel) {
            throw new Error('OLLAMA_URL or OLLAMA_VISION_MODEL not configured in environment');
        }
        const results = [];
        for (const screenshot of screenshots) {
            const formData = new form_data_1.default();
            formData.append('model', visionModel);
            formData.append('prompt', 'Extract all relevant financial and stock market information from this webpage screenshot. Include any news, analysis, stock prices, market data, and company information you can find.');
            formData.append('image', Buffer.from(screenshot, 'base64'), {
                filename: 'screenshot.png',
                contentType: 'image/png',
            });
            try {
                const response = await axios_1.default.post(`${ollamaUrl}/api/generate`, formData, {
                    headers: {
                        ...formData.getHeaders(),
                    },
                });
                if (response.data && response.data.response) {
                    results.push(response.data.response);
                }
            }
            catch (error) {
                console.error('Error analyzing screenshot with vision model:', error);
            }
        }
        return results.join('\n\n');
    }
    async scrapeUrl(url) {
        try {
            if (!this.driver) {
                await this.init();
            }
            await this.driver.get(url);
            // Wait for the page to load
            await this.driver.wait(selenium_webdriver_1.until.elementLocated(selenium_webdriver_1.By.css('body')), 10000);
            // Take screenshots while scrolling
            const screenshots = await this.takeFullPageScreenshot();
            // Analyze screenshots with vision model
            const content = await this.analyzeWithVisionModel(screenshots);
            return content;
        }
        catch (error) {
            console.error(`Error scraping ${url} with vision model:`, error);
            throw error;
        }
    }
    async close() {
        if (this.driver) {
            await this.driver.quit();
            this.driver = null;
        }
    }
}
exports.VisionScraper = VisionScraper;
