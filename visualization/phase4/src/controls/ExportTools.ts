/**
 * Export tools for LLMKG Phase 4 visualization
 * Provides comprehensive export capabilities for analysis and sharing
 */

export interface ScreenshotOptions {
  format: 'png' | 'jpg' | 'webp';
  quality: number; // 0-1 for jpg/webp
  width: number;
  height: number;
  scale: number; // DPI scaling
  transparent: boolean; // for PNG
  annotations: boolean;
  watermark: boolean;
  timestamp: boolean;
}

export interface VideoOptions {
  format: 'webm' | 'mp4';
  quality: 'low' | 'medium' | 'high' | 'lossless';
  fps: number;
  duration: number; // seconds
  bitrate?: number; // kbps
  codec: 'h264' | 'vp9' | 'av1';
  audio: boolean;
  annotations: boolean;
  watermark: boolean;
}

export interface DataExportOptions {
  format: 'json' | 'csv' | 'xlsx' | 'xml';
  includeMetadata: boolean;
  includeTimestamps: boolean;
  includeFilters: boolean;
  includeSettings: boolean;
  compressed: boolean;
  dateRange?: {
    start: Date;
    end: Date;
  };
  fields?: string[]; // specific fields to export
}

export interface ReportOptions {
  format: 'pdf' | 'html' | 'md';
  includeCharts: boolean;
  includeScreenshots: boolean;
  includeData: boolean;
  template: 'standard' | 'detailed' | 'executive' | 'technical';
  branding: boolean;
  interactive: boolean; // for HTML reports
}

export interface AnnotationData {
  id: string;
  type: 'text' | 'arrow' | 'highlight' | 'box';
  position: { x: number; y: number };
  content: string;
  style: {
    color: string;
    fontSize?: number;
    fontWeight?: string;
    backgroundColor?: string;
    borderColor?: string;
  };
  timestamp: Date;
}

class ExportToolsClass {
  private canvas: HTMLCanvasElement | null = null;
  private context: CanvasRenderingContext2D | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private recordingStream: MediaStream | null = null;
  private annotations: AnnotationData[] = [];

  constructor() {
    this.initializeCanvas();
  }

  private initializeCanvas(): void {
    this.canvas = document.createElement('canvas');
    this.context = this.canvas.getContext('2d');
  }

  // Screenshot functionality
  async captureScreenshot(
    element: HTMLElement,
    options: Partial<ScreenshotOptions> = {}
  ): Promise<Blob> {
    const defaultOptions: ScreenshotOptions = {
      format: 'png',
      quality: 0.95,
      width: element.offsetWidth,
      height: element.offsetHeight,
      scale: window.devicePixelRatio || 1,
      transparent: false,
      annotations: true,
      watermark: true,
      timestamp: true
    };

    const opts = { ...defaultOptions, ...options };

    // Use html2canvas for DOM element capture
    const html2canvas = await this.loadHtml2Canvas();
    const canvas = await html2canvas(element, {
      width: opts.width,
      height: opts.height,
      scale: opts.scale,
      backgroundColor: opts.transparent ? null : '#ffffff',
      useCORS: true,
      allowTaint: false,
      logging: false
    });

    // Add annotations if enabled
    if (opts.annotations && this.annotations.length > 0) {
      await this.addAnnotationsToCanvas(canvas);
    }

    // Add watermark if enabled
    if (opts.watermark) {
      await this.addWatermark(canvas);
    }

    // Add timestamp if enabled
    if (opts.timestamp) {
      await this.addTimestamp(canvas);
    }

    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        resolve(blob!);
      }, `image/${opts.format}`, opts.quality);
    });
  }

  private async loadHtml2Canvas(): Promise<any> {
    // Dynamically import html2canvas
    if (!(window as any).html2canvas) {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
      document.head.appendChild(script);
      
      return new Promise((resolve) => {
        script.onload = () => resolve((window as any).html2canvas);
      });
    }
    return (window as any).html2canvas;
  }

  private async addAnnotationsToCanvas(canvas: HTMLCanvasElement): Promise<void> {
    const ctx = canvas.getContext('2d')!;
    
    for (const annotation of this.annotations) {
      ctx.save();
      
      switch (annotation.type) {
        case 'text':
          ctx.fillStyle = annotation.style.color;
          ctx.font = `${annotation.style.fontWeight || 'normal'} ${annotation.style.fontSize || 16}px sans-serif`;
          ctx.fillText(annotation.content, annotation.position.x, annotation.position.y);
          break;
          
        case 'arrow':
          this.drawArrow(ctx, annotation);
          break;
          
        case 'highlight':
          ctx.fillStyle = annotation.style.backgroundColor || 'rgba(255, 255, 0, 0.3)';
          ctx.fillRect(annotation.position.x - 20, annotation.position.y - 10, 40, 20);
          break;
          
        case 'box':
          ctx.strokeStyle = annotation.style.borderColor || annotation.style.color;
          ctx.lineWidth = 2;
          ctx.strokeRect(annotation.position.x - 25, annotation.position.y - 15, 50, 30);
          break;
      }
      
      ctx.restore();
    }
  }

  private drawArrow(ctx: CanvasRenderingContext2D, annotation: AnnotationData): void {
    const { x, y } = annotation.position;
    const arrowSize = 10;
    
    ctx.strokeStyle = annotation.style.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - 20, y - 20);
    ctx.lineTo(x, y);
    ctx.lineTo(x - arrowSize, y - arrowSize);
    ctx.moveTo(x, y);
    ctx.lineTo(x - arrowSize, y + arrowSize);
    ctx.stroke();
  }

  private async addWatermark(canvas: HTMLCanvasElement): Promise<void> {
    const ctx = canvas.getContext('2d')!;
    ctx.save();
    ctx.globalAlpha = 0.3;
    ctx.fillStyle = '#666666';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('LLMKG Visualization', canvas.width - 20, canvas.height - 20);
    ctx.restore();
  }

  private async addTimestamp(canvas: HTMLCanvasElement): Promise<void> {
    const ctx = canvas.getContext('2d')!;
    const timestamp = new Date().toLocaleString();
    
    ctx.save();
    ctx.fillStyle = '#333333';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(timestamp, 20, 30);
    ctx.restore();
  }

  // Video recording functionality
  async startVideoRecording(
    element: HTMLElement,
    options: Partial<VideoOptions> = {}
  ): Promise<void> {
    const defaultOptions: VideoOptions = {
      format: 'webm',
      quality: 'high',
      fps: 30,
      duration: 60,
      codec: 'vp9',
      audio: false,
      annotations: true,
      watermark: true
    };

    const opts = { ...defaultOptions, ...options };

    try {
      // Capture the element as a media stream
      this.recordingStream = await (element as any).captureStream?.(opts.fps) ||
        await this.createStreamFromElement(element, opts.fps);

      if (!this.recordingStream) {
        throw new Error('Could not create media stream from element');
      }

      // Configure MediaRecorder
      const mimeType = this.getMimeType(opts.format, opts.codec);
      this.mediaRecorder = new MediaRecorder(this.recordingStream, {
        mimeType,
        videoBitsPerSecond: this.getBitrate(opts.quality)
      });

      const chunks: Blob[] = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: mimeType });
        this.downloadBlob(blob, `llmkg-recording-${Date.now()}.${opts.format}`);
        this.cleanup();
      };

      this.mediaRecorder.start();

      // Auto-stop after duration
      setTimeout(() => {
        this.stopVideoRecording();
      }, opts.duration * 1000);

    } catch (error) {
      console.error('Error starting video recording:', error);
      throw error;
    }
  }

  async stopVideoRecording(): Promise<void> {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.stop();
    }
  }

  private async createStreamFromElement(element: HTMLElement, fps: number): Promise<MediaStream> {
    // Fallback: use canvas capture
    const canvas = document.createElement('canvas');
    canvas.width = element.offsetWidth;
    canvas.height = element.offsetHeight;
    
    const ctx = canvas.getContext('2d')!;
    
    // Capture element to canvas at specified FPS
    const interval = setInterval(async () => {
      const tempCanvas = await this.captureElementToCanvas(element);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(tempCanvas, 0, 0);
    }, 1000 / fps);

    const stream = canvas.captureStream(fps);
    
    // Clean up interval when stream ends
    stream.getVideoTracks()[0].addEventListener('ended', () => {
      clearInterval(interval);
    });

    return stream;
  }

  private async captureElementToCanvas(element: HTMLElement): Promise<HTMLCanvasElement> {
    const html2canvas = await this.loadHtml2Canvas();
    return html2canvas(element, {
      logging: false,
      useCORS: true
    });
  }

  private getMimeType(format: VideoOptions['format'], codec: VideoOptions['codec']): string {
    const mimeTypes = {
      webm: {
        vp9: 'video/webm; codecs=vp9',
        h264: 'video/webm; codecs=h264',
        av1: 'video/webm; codecs=av01'
      },
      mp4: {
        h264: 'video/mp4; codecs=h264',
        vp9: 'video/mp4; codecs=vp9',
        av1: 'video/mp4; codecs=av01'
      }
    };

    return mimeTypes[format][codec] || 'video/webm';
  }

  private getBitrate(quality: VideoOptions['quality']): number {
    const bitrates = {
      low: 1000000,      // 1 Mbps
      medium: 2500000,   // 2.5 Mbps
      high: 5000000,     // 5 Mbps
      lossless: 20000000 // 20 Mbps
    };
    
    return bitrates[quality];
  }

  // Data export functionality
  async exportData(data: any[], options: Partial<DataExportOptions> = {}): Promise<Blob> {
    const defaultOptions: DataExportOptions = {
      format: 'json',
      includeMetadata: true,
      includeTimestamps: true,
      includeFilters: false,
      includeSettings: false,
      compressed: false
    };

    const opts = { ...defaultOptions, ...options };

    // Filter data by date range if specified
    let filteredData = data;
    if (opts.dateRange) {
      filteredData = data.filter(item => {
        const timestamp = new Date(item.timestamp || item.created_at || item.time);
        return timestamp >= opts.dateRange!.start && timestamp <= opts.dateRange!.end;
      });
    }

    // Select specific fields if specified
    if (opts.fields && opts.fields.length > 0) {
      filteredData = filteredData.map(item => {
        const filtered: any = {};
        for (const field of opts.fields!) {
          if (item.hasOwnProperty(field)) {
            filtered[field] = item[field];
          }
        }
        return filtered;
      });
    }

    // Add metadata if requested
    const exportData = {
      data: filteredData,
      ...(opts.includeMetadata && {
        metadata: {
          exported: new Date().toISOString(),
          totalRecords: filteredData.length,
          format: opts.format,
          ...(opts.dateRange && {
            dateRange: {
              start: opts.dateRange.start.toISOString(),
              end: opts.dateRange.end.toISOString()
            }
          })
        }
      })
    };

    let blob: Blob;

    switch (opts.format) {
      case 'json':
        const jsonString = JSON.stringify(exportData, null, 2);
        blob = new Blob([jsonString], { type: 'application/json' });
        break;

      case 'csv':
        const csvString = this.convertToCSV(filteredData);
        blob = new Blob([csvString], { type: 'text/csv' });
        break;

      case 'xlsx':
        blob = await this.convertToXLSX(filteredData);
        break;

      case 'xml':
        const xmlString = this.convertToXML(exportData);
        blob = new Blob([xmlString], { type: 'application/xml' });
        break;

      default:
        throw new Error(`Unsupported export format: ${opts.format}`);
    }

    if (opts.compressed) {
      blob = await this.compressBlob(blob);
    }

    return blob;
  }

  private convertToCSV(data: any[]): string {
    if (data.length === 0) return '';

    const headers = Object.keys(data[0]);
    const csvRows = [headers.join(',')];

    for (const row of data) {
      const values = headers.map(header => {
        const value = row[header];
        if (value === null || value === undefined) return '';
        if (typeof value === 'string' && value.includes(',')) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return String(value);
      });
      csvRows.push(values.join(','));
    }

    return csvRows.join('\n');
  }

  private async convertToXLSX(data: any[]): Promise<Blob> {
    // Use SheetJS for XLSX conversion
    const XLSX = await this.loadXLSX();
    const worksheet = XLSX.utils.json_to_sheet(data);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Data');
    
    const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
    return new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
  }

  private async loadXLSX(): Promise<any> {
    if (!(window as any).XLSX) {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js';
      document.head.appendChild(script);
      
      return new Promise((resolve) => {
        script.onload = () => resolve((window as any).XLSX);
      });
    }
    return (window as any).XLSX;
  }

  private convertToXML(data: any): string {
    const xmlHeader = '<?xml version="1.0" encoding="UTF-8"?>\n';
    const xmlData = this.objectToXML(data, 'export');
    return xmlHeader + xmlData;
  }

  private objectToXML(obj: any, rootName: string): string {
    if (typeof obj !== 'object') {
      return `<${rootName}>${this.escapeXML(String(obj))}</${rootName}>`;
    }

    if (Array.isArray(obj)) {
      return obj.map(item => this.objectToXML(item, 'item')).join('\n');
    }

    const xmlParts = [`<${rootName}>`];
    for (const [key, value] of Object.entries(obj)) {
      if (Array.isArray(value)) {
        xmlParts.push(`<${key}>`);
        xmlParts.push(this.objectToXML(value, 'item'));
        xmlParts.push(`</${key}>`);
      } else if (typeof value === 'object' && value !== null) {
        xmlParts.push(this.objectToXML(value, key));
      } else {
        xmlParts.push(`<${key}>${this.escapeXML(String(value))}</${key}>`);
      }
    }
    xmlParts.push(`</${rootName}>`);

    return xmlParts.join('\n');
  }

  private escapeXML(str: string): string {
    return str.replace(/[<>&'"]/g, (char) => {
      const entities: { [key: string]: string } = {
        '<': '&lt;',
        '>': '&gt;',
        '&': '&amp;',
        "'": '&apos;',
        '"': '&quot;'
      };
      return entities[char];
    });
  }

  private async compressBlob(blob: Blob): Promise<Blob> {
    const stream = new CompressionStream('gzip');
    const compressedStream = blob.stream().pipeThrough(stream);
    return new Response(compressedStream).blob();
  }

  // Report generation
  async generateReport(
    data: any[],
    visualizations: Blob[],
    options: Partial<ReportOptions> = {}
  ): Promise<Blob> {
    const defaultOptions: ReportOptions = {
      format: 'pdf',
      includeCharts: true,
      includeScreenshots: true,
      includeData: true,
      template: 'standard',
      branding: true,
      interactive: false
    };

    const opts = { ...defaultOptions, ...options };

    switch (opts.format) {
      case 'html':
        return this.generateHTMLReport(data, visualizations, opts);
      case 'pdf':
        return this.generatePDFReport(data, visualizations, opts);
      case 'md':
        return this.generateMarkdownReport(data, visualizations, opts);
      default:
        throw new Error(`Unsupported report format: ${opts.format}`);
    }
  }

  private async generateHTMLReport(data: any[], visualizations: Blob[], options: ReportOptions): Promise<Blob> {
    const template = this.getHTMLTemplate(options.template);
    const reportData = {
      title: 'LLMKG Visualization Report',
      generated: new Date().toISOString(),
      summary: this.generateSummary(data),
      data: options.includeData ? data.slice(0, 100) : [], // Limit data for HTML
      charts: options.includeCharts,
      visualizations: options.includeScreenshots ? visualizations : []
    };

    const html = this.renderHTMLTemplate(template, reportData);
    return new Blob([html], { type: 'text/html' });
  }

  private async generatePDFReport(data: any[], visualizations: Blob[], options: ReportOptions): Promise<Blob> {
    // Generate HTML first, then convert to PDF
    const htmlBlob = await this.generateHTMLReport(data, visualizations, options);
    const html = await htmlBlob.text();
    
    // Use jsPDF for PDF generation
    const jsPDF = await this.loadJsPDF();
    const doc = new jsPDF();
    
    // Add content to PDF (simplified)
    doc.text('LLMKG Visualization Report', 20, 20);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 30);
    
    if (options.includeData) {
      doc.text('Data Summary:', 20, 50);
      doc.text(`Total Records: ${data.length}`, 20, 60);
    }

    return new Blob([doc.output('blob')], { type: 'application/pdf' });
  }

  private async generateMarkdownReport(data: any[], visualizations: Blob[], options: ReportOptions): Promise<Blob> {
    const lines = [
      '# LLMKG Visualization Report',
      '',
      `Generated: ${new Date().toLocaleString()}`,
      '',
      '## Summary',
      '',
      `- Total Records: ${data.length}`,
      `- Date Range: ${this.getDateRange(data)}`,
      `- Export Format: Markdown`,
      ''
    ];

    if (options.includeData) {
      lines.push('## Data Overview', '');
      const summary = this.generateSummary(data);
      lines.push(`Average Processing Time: ${summary.avgProcessingTime}ms`);
      lines.push(`Success Rate: ${summary.successRate}%`);
      lines.push('');
    }

    if (options.includeScreenshots) {
      lines.push('## Visualizations', '');
      lines.push(`${visualizations.length} visualization(s) included`);
      lines.push('');
    }

    const markdown = lines.join('\n');
    return new Blob([markdown], { type: 'text/markdown' });
  }

  private async loadJsPDF(): Promise<any> {
    if (!(window as any).jsPDF) {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
      document.head.appendChild(script);
      
      return new Promise((resolve) => {
        script.onload = () => resolve((window as any).jsPDF.jsPDF);
      });
    }
    return (window as any).jsPDF.jsPDF;
  }

  private getHTMLTemplate(template: ReportOptions['template']): string {
    // Simplified HTML template
    return `
      <!DOCTYPE html>
      <html>
        <head>
          <title>LLMKG Report</title>
          <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { border-bottom: 2px solid #333; padding-bottom: 20px; }
            .summary { background: #f5f5f5; padding: 20px; margin: 20px 0; }
            .data-table { width: 100%; border-collapse: collapse; }
            .data-table th, .data-table td { border: 1px solid #ddd; padding: 8px; }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>{{title}}</h1>
            <p>Generated: {{generated}}</p>
          </div>
          <div class="summary">
            <h2>Summary</h2>
            <p>Total Records: {{summary.totalRecords}}</p>
            <p>Success Rate: {{summary.successRate}}%</p>
          </div>
          {{#if includeData}}
          <div class="data">
            <h2>Data Sample</h2>
            <table class="data-table">
              <thead>
                <tr>
                  {{#each dataHeaders}}
                  <th>{{this}}</th>
                  {{/each}}
                </tr>
              </thead>
              <tbody>
                {{#each data}}
                <tr>
                  {{#each this}}
                  <td>{{this}}</td>
                  {{/each}}
                </tr>
                {{/each}}
              </tbody>
            </table>
          </div>
          {{/if}}
        </body>
      </html>
    `;
  }

  private renderHTMLTemplate(template: string, data: any): string {
    // Simple template rendering (replace {{}} placeholders)
    return template.replace(/\{\{([^}]+)\}\}/g, (match, key) => {
      const value = this.getNestedValue(data, key.trim());
      return value !== undefined ? String(value) : match;
    });
  }

  private getNestedValue(obj: any, path: string): any {
    return path.split('.').reduce((current, key) => current?.[key], obj);
  }

  private generateSummary(data: any[]): any {
    if (data.length === 0) {
      return {
        totalRecords: 0,
        avgProcessingTime: 0,
        successRate: 0
      };
    }

    const processingTimes = data
      .filter(item => item.duration || item.processing_time)
      .map(item => item.duration || item.processing_time);

    const successfulItems = data.filter(item => 
      item.status === 'success' || item.success === true || item.error === undefined
    );

    return {
      totalRecords: data.length,
      avgProcessingTime: processingTimes.length > 0 
        ? Math.round(processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length)
        : 0,
      successRate: Math.round((successfulItems.length / data.length) * 100)
    };
  }

  private getDateRange(data: any[]): string {
    if (data.length === 0) return 'No data';

    const timestamps = data
      .map(item => new Date(item.timestamp || item.created_at || item.time))
      .filter(date => !isNaN(date.getTime()))
      .sort((a, b) => a.getTime() - b.getTime());

    if (timestamps.length === 0) return 'No timestamps';

    const start = timestamps[0].toLocaleDateString();
    const end = timestamps[timestamps.length - 1].toLocaleDateString();
    
    return start === end ? start : `${start} - ${end}`;
  }

  // Annotation system
  addAnnotation(annotation: Omit<AnnotationData, 'id' | 'timestamp'>): AnnotationData {
    const newAnnotation: AnnotationData = {
      ...annotation,
      id: crypto.randomUUID(),
      timestamp: new Date()
    };

    this.annotations.push(newAnnotation);
    return newAnnotation;
  }

  removeAnnotation(id: string): boolean {
    const index = this.annotations.findIndex(a => a.id === id);
    if (index >= 0) {
      this.annotations.splice(index, 1);
      return true;
    }
    return false;
  }

  clearAnnotations(): void {
    this.annotations = [];
  }

  getAnnotations(): AnnotationData[] {
    return [...this.annotations];
  }

  // Utility methods
  private downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  private cleanup(): void {
    if (this.recordingStream) {
      this.recordingStream.getTracks().forEach(track => track.stop());
      this.recordingStream = null;
    }
    
    if (this.mediaRecorder) {
      this.mediaRecorder = null;
    }
  }

  // Public download methods
  async downloadScreenshot(element: HTMLElement, options?: Partial<ScreenshotOptions>): Promise<void> {
    const blob = await this.captureScreenshot(element, options);
    const filename = `llmkg-screenshot-${Date.now()}.${options?.format || 'png'}`;
    this.downloadBlob(blob, filename);
  }

  async downloadData(data: any[], options?: Partial<DataExportOptions>): Promise<void> {
    const blob = await this.exportData(data, options);
    const filename = `llmkg-data-${Date.now()}.${options?.format || 'json'}`;
    this.downloadBlob(blob, filename);
  }

  async downloadReport(data: any[], visualizations: Blob[], options?: Partial<ReportOptions>): Promise<void> {
    const blob = await this.generateReport(data, visualizations, options);
    const filename = `llmkg-report-${Date.now()}.${options?.format || 'pdf'}`;
    this.downloadBlob(blob, filename);
  }
}

// Singleton instance
export const exportTools = new ExportToolsClass();