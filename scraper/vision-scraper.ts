import { Builder, By, until, WebDriver } from 'selenium-webdriver';
import { Options as ChromeOptions } from 'selenium-webdriver/chrome';
import * as fs from 'fs/promises';
import * as path from 'path';
import axios, { AxiosError } from 'axios';
import FormData from 'form-data';
import * as dotenv from 'dotenv';
import 'chromedriver';

dotenv.config();

// Log environment variables
console.log('Environment variables:', {
  OLLAMA_URL: process.env.OLLAMA_URL || 'not set',
  OLLAMA_VISION_MODEL: process.env.OLLAMA_VISION_MODEL || 'not set'
});

interface VisionScraperOptions {
  screenshotDir?: string;
  scrollDelay?: number;
  maxScrolls?: number;
  pageLoadTimeout?: number;
}

interface VisionResult {
    stockInfo: Record<string, string>;
    keyStats: Record<string, string>;
    news: string[];
    notableEvents: string[];
    technicalIndicators: string[];
}

export class VisionScraper {
  private driver: WebDriver | null = null;
  private options: Required<VisionScraperOptions>;

  constructor(options: VisionScraperOptions = {}) {
    this.options = {
      screenshotDir: path.join(process.cwd(), 'screenshots'),
      scrollDelay: 2000, // Increased delay to allow more time for dynamic content
      maxScrolls: 5,
      pageLoadTimeout: 30000, // 30 seconds timeout for page load
      ...options
    };
  }

  async init() {
    console.log('Initializing vision scraper...');
    const chromeOptions = new ChromeOptions();
    chromeOptions.addArguments(
        '--headless=new',
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--window-size=1920,1080',
        '--start-maximized',
        '--disable-blink-features=AutomationControlled',
        '--disable-web-security',
        '--ignore-certificate-errors',
        '--disable-software-rasterizer',  // Add this
        '--disable-webgl',                // Add this
        '--ignore-ssl-errors=yes',        // Add this
        '--ignore-certificate-errors-spki-list', // Add this
        '--allow-running-insecure-content'  // Add this
    );

    // Add proxy if configured
    const proxy = process.env.PROXY_SERVER;
    if (proxy) {
      console.log('Using proxy server:', proxy);
      chromeOptions.addArguments(`--proxy-server=${proxy}`);
    }

    // Add stealth settings
    chromeOptions.addArguments(
      `--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36`
    );

    this.driver = await new Builder()
      .forBrowser('chrome')
      .setChromeOptions(chromeOptions)
      .build();

    // Set page load timeout
    await this.driver.manage().setTimeouts({ pageLoad: this.options.pageLoadTimeout });
    console.log('Chrome driver initialized successfully');

    // Ensure screenshot directory exists
    await fs.mkdir(this.options.screenshotDir, { recursive: true });
    console.log('Screenshot directory created at:', this.options.screenshotDir);

    // Set stealth properties
    await this.driver.executeScript(`
      Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined
      });
    `);
  }

  private async saveScreenshot(screenshot: string, index: number): Promise<void> {
    const filename = path.join(this.options.screenshotDir, `screenshot_${index}.png`);
    await fs.writeFile(filename, screenshot, 'base64');
    console.log(`Screenshot saved to: ${filename}`);
  }

  private async waitForContentToLoad(): Promise<void> {
    try {
      // Wait for basic page load
      await this.driver!.wait(until.elementLocated(By.css('body')), 10000);
      
      // Enhanced waiting for dynamic content
      const waitForSelectors = [
        // Common article/content selectors
        'article', '.article', '.content', '.main-content', 
        // News specific selectors
        '.news-content', '.story-content', '.article-body',
        // Financial data selectors
        '.stock-data', '.market-data', '.quote-data',
        // Generic content selectors
        '[role="main"]', 'main', '#main-content'
      ];

      for (const selector of waitForSelectors) {
        try {
          await this.driver!.wait(until.elementLocated(By.css(selector)), 2000);
          console.log(`Found content with selector: ${selector}`);
          break;
        } catch {
          continue;
        }
      }

      // Wait for any loading spinners to disappear
      const spinnerSelectors = ['.loading', '.spinner', '.loader', '[role="progressbar"]'];
      for (const selector of spinnerSelectors) {
        try {
          const spinners = await this.driver!.findElements(By.css(selector));
          if (spinners.length > 0) {
            await this.driver!.wait(until.elementIsNotVisible(spinners[0]), 5000);
          }
        } catch (error) {
          // Ignore errors if spinner elements are not found
        }
      }

      // Wait for dynamic content
      await this.driver!.sleep(2000);

      // Wait for any lazy-loaded images
      await this.driver!.executeScript(`
        return new Promise((resolve) => {
          const images = document.getElementsByTagName('img');
          const loadedImages = Array.from(images).filter(img => !img.complete);
          
          if (loadedImages.length === 0) {
            resolve(true);
          }
          
          const promises = loadedImages.map(img => {
            return new Promise((resolve) => {
              img.addEventListener('load', resolve);
              img.addEventListener('error', resolve);
            });
          });
          
          Promise.all(promises).then(() => resolve(true));
          
          // Fallback timeout
          setTimeout(resolve, 5000);
        });
      `);

    } catch (error) {
      console.warn('Warning: Some content might not have loaded:', error instanceof Error ? error.message : error);
    }
  }

  private async bypassAntiBot(): Promise<void> {
    try {
      // Add random mouse movements
      const actions = this.driver!.actions();
      for (let i = 0; i < 3; i++) {
        const x = Math.floor(Math.random() * 500);
        const y = Math.floor(Math.random() * 500);
        await actions.move({x, y}).perform();
        await this.driver!.sleep(Math.random() * 1000);
      }

      // Scroll smoothly
      await this.driver!.executeScript(`
        window.scrollTo({
          top: Math.floor(Math.random() * 500),
          behavior: 'smooth'
        });
      `);

      await this.driver!.sleep(1000);
    } catch (error) {
      console.warn('Warning: Error during anti-bot bypass:', error instanceof Error ? error.message : error);
    }
  }

  private async takeFullPageScreenshot(): Promise<string[]> {
    if (!this.driver) throw new Error('Driver not initialized');

    console.log('Taking full page screenshots...');
    const screenshots: string[] = [];
    let lastHeight = 0;
    let scrollCount = 0;

    try {
      // Set initial window size
      await this.driver.manage().window().setRect({ width: 1920, height: 1080 });
      
      // Wait for content to load
      await this.waitForContentToLoad();
      
      // Get the full height of the page
      const fullHeight = await this.driver.executeScript('return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);') as number;
      console.log(`Full page height: ${fullHeight}px`);

      // First, try to find and screenshot the main content area
      const mainContentSelectors = [
        '.main-content',
        '#main-content',
        '[role="main"]',
        '.stock-content',
        '.quote-content',
        '.market-data'
      ];

      for (const selector of mainContentSelectors) {
        try {
          const mainContent = await this.driver.findElement(By.css(selector));
          if (mainContent) {
            // Take screenshot of main content area
            const mainContentScreenshot = await mainContent.takeScreenshot();
            if (mainContentScreenshot) {
              await this.saveScreenshot(mainContentScreenshot, 0);
              screenshots.push(mainContentScreenshot);
              console.log(`Captured main content area using selector: ${selector}`);
            }
          }
        } catch (error) {
          // Continue to next selector if element not found
        }
      }

      // If no main content area found, proceed with full page screenshots
      if (screenshots.length === 0) {
        while (scrollCount < this.options.maxScrolls) {
          // Wait for any dynamic content to load after scroll
          await this.driver.sleep(this.options.scrollDelay);
          await this.waitForContentToLoad();

          // Take screenshot of current viewport
          const screenshot = await this.driver.takeScreenshot().catch(err => {
            console.log('Screenshot warning (non-critical):', err.message);
            return null;
          });
          if (!screenshot) {
            console.log('Failed to capture screenshot, continuing...');
            continue;
          }

          // Save the screenshot to file
          await this.saveScreenshot(screenshot, scrollCount);
          screenshots.push(screenshot);
          console.log(`Took screenshot ${scrollCount + 1}/${this.options.maxScrolls}`);

          // Get current scroll position
          const currentHeight = await this.driver.executeScript('return window.pageYOffset + window.innerHeight;') as number;
          const remainingScroll = fullHeight - currentHeight;

          // If we've reached the bottom, break
          if (currentHeight === lastHeight || remainingScroll <= 0) {
            console.log('Reached bottom of page');
            break;
          }

          // Scroll by viewport height
          await this.driver.executeScript('window.scrollBy(0, window.innerHeight);');
          lastHeight = currentHeight;
          scrollCount++;
        }
      }

      console.log(`Captured ${screenshots.length} screenshots`);
      return screenshots;

    } catch (error) {
      console.error('Error taking screenshots:', error instanceof Error ? error.message : error);
      throw error;
    }
  }

  private async analyzeWithVisionModel(screenshots: string[]): Promise<string> {
    console.log('Analyzing screenshots with vision model...');
    // Remove /v1 from URL and ensure no trailing slash
    const ollamaUrl = (process.env.OLLAMA_URL || '').replace(/\/v1\/?$/, '').replace(/\/$/, '');
    // Get model name from environment variable with fallback
    const visionModel = process.env.OLLAMA_VISION_MODEL || 'llama3.2-vision:11b-instruct-q8_0';
    const maxRetries = 3;
    const retryDelay = 5000; // 5 seconds

    if (!ollamaUrl) {
        throw new Error('OLLAMA_URL not configured in environment');
    }

    console.log('Using Ollama configuration:', {
        url: ollamaUrl,
        model: visionModel
    });

    const results: any[] = [];

    for (const [index, screenshot] of screenshots.entries()) {
        console.log(`Processing screenshot ${index + 1}/${screenshots.length}`);
        
        let retryCount = 0;
        while (retryCount < maxRetries) {
            try {
                const requestBody = {
                    model: visionModel,
                    messages: [{
                        role: 'user',
                        content: 'What financial information can you see in this image? Please extract any stock prices, market data, news headlines, or technical indicators.',
                        images: [screenshot]
                    }],
                    stream: true
                };

                console.log('Sending request to Ollama vision model...');
                const response = await axios.post(`${ollamaUrl}/api/chat`, requestBody, {
                    timeout: 120000, // 2 minute timeout for vision processing
                    maxBodyLength: Infinity,
                    maxContentLength: Infinity,
                    responseType: 'stream'
                });

                let fullContent = '';
                
                // Process streaming response
                await new Promise((resolve, reject) => {
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

                    response.data.on('end', () => {
                        console.log('\nFinished processing vision response');
                        resolve(fullContent);
                    });

                    response.data.on('error', (error: Error) => {
                        console.error('Error in vision response stream:', error);
                        reject(error);
                    });
                });

                if (fullContent) {
                    try {
                        // Try to parse as structured data
                        const parsedContent = this.parseVisionResponse(fullContent);
                        results.push(parsedContent);
                    } catch (parseError) {
                        // If parsing fails, store raw content
                        results.push(fullContent);
                    }
                    break;
                } else {
                    throw new Error('No content received from vision model');
                }

            } catch (error) {
                console.error('Error from vision model:', error instanceof Error ? error.message : error);
                retryCount++;
                if (retryCount === maxRetries) {
                    console.error('Final attempt failed analyzing screenshot');
                    throw error;
                } else {
                    console.log(`Attempt ${retryCount} failed, retrying in ${retryDelay}ms...`);
                    await new Promise(resolve => setTimeout(resolve, retryDelay));
                }
            }
        }
    }

    // Combine results into a single structured output
    const combinedResults = this.combineVisionResults(results);
    return JSON.stringify(combinedResults, null, 2);
  }

  private parseVisionResponse(content: string): VisionResult {
    try {
        // First try to parse as JSON
        return JSON.parse(content) as VisionResult;
    } catch {
        // If not JSON, try to extract structured data from text
        const structured: VisionResult = {
            stockInfo: {},
            keyStats: {},
            news: [],
            notableEvents: [],
            technicalIndicators: []
        };

        // Basic text parsing logic
        const lines = content.split('\n');
        let currentSection = '';
        
        for (const line of lines) {
            if (line.includes('Stock Information:')) {
                currentSection = 'stockInfo';
            } else if (line.includes('Key Statistics:')) {
                currentSection = 'keyStats';
            } else if (line.includes('Recent News Headlines:')) {
                currentSection = 'news';
            } else if (line.includes('Notable Events:')) {
                currentSection = 'notableEvents';
            } else if (line.includes('Technical Indicators:')) {
                currentSection = 'technicalIndicators';
            } else if (line.trim()) {
                // Add non-empty lines to appropriate section
                if (currentSection === 'news' || currentSection === 'notableEvents' || currentSection === 'technicalIndicators') {
                    (structured[currentSection as keyof VisionResult] as string[]).push(line.trim());
                } else if (currentSection && (currentSection === 'stockInfo' || currentSection === 'keyStats')) {
                    const [key, value] = line.split(':').map(s => s.trim());
                    if (key && value) {
                        (structured[currentSection] as Record<string, string>)[key] = value;
                    }
                }
            }
        }

        return structured;
    }
  }

  private combineVisionResults(results: (VisionResult | string)[]): VisionResult {
    return results.reduce((acc: VisionResult, curr) => {
        if (typeof curr === 'string') {
            return acc;
        }
        return {
            stockInfo: { ...acc.stockInfo, ...curr.stockInfo },
            keyStats: { ...acc.keyStats, ...curr.keyStats },
            news: [...acc.news, ...curr.news].filter((v, i, a) => a.indexOf(v) === i),
            notableEvents: [...acc.notableEvents, ...curr.notableEvents].filter((v, i, a) => a.indexOf(v) === i),
            technicalIndicators: [...acc.technicalIndicators, ...curr.technicalIndicators].filter((v, i, a) => a.indexOf(v) === i)
        };
    }, {
        stockInfo: {},
        keyStats: {},
        news: [],
        notableEvents: [],
        technicalIndicators: []
    });
  }

  async scrapeUrl(url: string): Promise<string> {
    try {
      console.log(`Starting to scrape ${url} with vision model...`);
      if (!this.driver) {
        await this.init();
      }

      // Add random delay before loading page
      await this.driver!.sleep(Math.floor(Math.random() * 2000) + 1000);

      console.log('Navigating to URL...');
      await this.driver!.get(url);
      
      // Bypass anti-bot measures
      await this.bypassAntiBot();
      
      // Wait for the page to load
      console.log('Waiting for page to load...');
      await this.waitForContentToLoad();
      
      // First attempt: Vision scraping
      try {
        console.log('Attempting vision scraping...');
        const screenshots = await this.takeFullPageScreenshot();
        const visionContent = await this.analyzeWithVisionModel(screenshots);
        
        // Validate if we got meaningful content from vision scraping
        const parsedContent = JSON.parse(visionContent);
        if (
          parsedContent.stockInfo?.currentPrice || 
          parsedContent.keyStats?.peRatio || 
          parsedContent.news?.length > 0
        ) {
          console.log('Vision scraping successful');
          return visionContent;
        }
        
        throw new Error('Vision scraping did not yield meaningful results');
      } catch (visionError) {
        console.log('Vision scraping failed or yielded insufficient results:', visionError instanceof Error ? visionError.message : visionError);
        
        // Fallback: Regular scraping with enhanced selectors
        console.log('Falling back to regular scraping...');
        const content = await this.fallbackScraping();
        if (content) {
          return JSON.stringify(content);
        }
        throw new Error('Both vision and regular scraping failed');
      }
    } catch (error) {
      console.error(`Error scraping ${url}:`, error instanceof Error ? error.message : error);
      throw error;
    }
  }

  private async fallbackScraping(): Promise<any> {
    const data: any = {
      stockInfo: {},
      keyStats: {},
      news: [],
      notableEvents: [],
      technicalIndicators: []
    };

    try {
      // Enhanced selectors for financial data
      const selectors = {
        price: ['[data-test="qsp-price"]', '.price', '.stock-price', '[data-symbol]'],
        priceChange: ['[data-test="qsp-price-change"]', '.price-change', '.stock-change'],
        volume: ['[data-test="qsp-volume"]', '.volume', '.trading-volume'],
        marketCap: ['[data-test="qsp-market-cap"]', '.market-cap', '.company-value'],
        peRatio: ['[data-test="qsp-pe-ratio"]', '.pe-ratio', '.price-earnings'],
        eps: ['[data-test="qsp-eps"]', '.eps', '.earnings-per-share'],
        dividend: ['[data-test="qsp-dividend-yield"]', '.dividend-yield', '.dividend']
      };

      // Try each selector
      for (const [key, selectorList] of Object.entries(selectors)) {
        for (const selector of selectorList) {
          try {
            const element = await this.driver!.findElement(By.css(selector));
            const text = await element.getText();
            if (text) {
              data.stockInfo[key] = text;
              break;
            }
          } catch {
            continue;
          }
        }
      }

      // Get news headlines
      const newsSelectors = [
        '[data-test="news-item"]', 
        '.news-headline', 
        '.article-headline',
        '.news-title'
      ];

      for (const selector of newsSelectors) {
        try {
          const newsElements = await this.driver!.findElements(By.css(selector));
          for (const element of newsElements.slice(0, 5)) {
            const text = await element.getText();
            if (text) {
              data.news.push(text);
            }
          }
          if (data.news.length > 0) break;
        } catch {
          continue;
        }
      }

      return data;
    } catch (error) {
      console.error('Error in fallback scraping:', error instanceof Error ? error.message : error);
      return null;
    }
  }

  async close() {
    if (this.driver) {
      console.log('Closing Chrome driver...');
      await this.driver.quit();
      this.driver = null;
      console.log('Chrome driver closed');
    }
  }
} 