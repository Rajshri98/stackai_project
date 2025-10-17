// Minimal state and cached DOM references
let uploadedFiles = [];
let isUploading = false;
let chatMessages = [];

// DOM elements
let fileInput;
let uploadArea;
let uploadBtn;
let chatInterface;
let chatMessagesContainer;
let queryInput;
let sendBtn;
let uploadStatus;
let progressContainer;
let progressBar;
let progressText;

// Initialize UI and wire up events
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    loadTheme();
    loadUIVariant();
    updateUI();
    
    // Check if documents are already loaded
    checkForExistingDocuments();
});

// Event listeners and element resolution
function initializeEventListeners() {
    // Resolve elements now that DOM is ready
    fileInput = document.getElementById('fileInput');
    uploadArea = document.querySelector('.upload-area');
    // Support either a dedicated .upload-btn or the generic button inside upload area
    uploadBtn = document.querySelector('.upload-btn') || document.querySelector('.upload-area .btn');
    chatInterface = document.querySelector('.chat-interface');
    chatMessagesContainer = document.getElementById('chatMessages');
    queryInput = document.getElementById('queryInput');
    sendBtn = document.getElementById('sendBtn');
    uploadStatus = document.getElementById('uploadStatus');
    progressContainer = document.querySelector('.progress-container');
    progressBar = document.querySelector('.progress-bar');
    progressText = document.querySelector('.progress-text');

    // File input handling
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelection);
        fileInput.addEventListener('input', handleFileSelection);
    }
    
    // Drag and drop handling
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
        uploadArea.addEventListener('click', () => fileInput && fileInput.click());
    }
    
    // Upload button
    if (uploadBtn) {
        uploadBtn.addEventListener('click', (e) => {
            // Prevent the upload-area click from also firing
            if (e && e.stopPropagation) e.stopPropagation();
            uploadFiles();
        });
    }
    
    // Chat interface
    if (queryInput) queryInput.addEventListener('keypress', handleKeyPress);
    if (sendBtn) sendBtn.addEventListener('click', askQuestion);
    
    // Quick action buttons
    document.querySelectorAll('.quick-action').forEach(action => {
        action.addEventListener('click', (e) => {
            const question = e.currentTarget.getAttribute('data-question');
            if (question) {
                askQuickQuestion(question);
            }
        });
    });
    
    // Theme toggle
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle) themeToggle.addEventListener('click', toggleTheme);

    // UI variant toggle
    const uiToggle = document.querySelector('.ui-toggle');
    if (uiToggle) {
        uiToggle.addEventListener('click', toggleUIVariant);
    }
}

// File selection/drag handlers (auto-upload)
function handleFileSelection(e) {
    console.log('File input changed, files:', e.target.files);
    uploadedFiles = Array.from(e.target.files);
    console.log('Uploaded files set to:', uploadedFiles);
    updateFileInfo();
    // Auto-start upload after selection
    if (uploadedFiles && uploadedFiles.length > 0) {
        setTimeout(() => uploadFiles(), 50);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    uploadedFiles = Array.from(e.dataTransfer.files).filter(file => file.type === 'application/pdf');
    fileInput.files = e.dataTransfer.files;
    updateFileInfo();
    // Auto-start upload after drop
    if (uploadedFiles && uploadedFiles.length > 0) {
        setTimeout(() => uploadFiles(), 50);
    }
}

function updateFileInfo() {
    const fileInfo = document.getElementById('fileInfo');
    if (uploadedFiles.length > 0) {
        fileInfo.innerHTML = `
            <strong>Selected ${uploadedFiles.length} file(s):</strong><br>
            ${uploadedFiles.map(f => f.name).join('<br>')}
        `;
    } else {
        fileInfo.innerHTML = '';
    }
}

// Upload selected files to the backend
async function uploadFiles() {
    console.log('Upload files called, uploadedFiles:', uploadedFiles);
    if (isUploading) {
        console.log('Upload already in progress, skipping duplicate call');
        return;
    }
    // Fallback: read from input if global array is empty
    if ((!uploadedFiles || uploadedFiles.length === 0) && fileInput && fileInput.files && fileInput.files.length > 0) {
        uploadedFiles = Array.from(fileInput.files);
    }
    
    if (uploadedFiles.length === 0) {
        showStatus('Please select at least one PDF file.', 'error');
        return;
    }

    // Clear any previous status
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.innerHTML = '';
    
    // Show progress
    showProgress();
    showStatus('Processing documents...', 'loading');

    try {
        isUploading = true;
        const formData = new FormData();
        uploadedFiles.forEach(file => {
            console.log('Adding file to FormData:', file.name, file.type, file.size);
            formData.append('files', file);
        });

        console.log('Sending request to /ingest...');
        const response = await fetch('/ingest', {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status);
        const result = await response.json();
        console.log('Response result:', result);

        if (response.ok) {
            hideProgress();
            showStatus(`Successfully processed ${result.chunks_created} text chunks from your documents`, 'success');
            addMessage('assistant', `Great! I've processed ${result.chunks_created} chunks from your documents. You can now ask me questions about them.`);
            console.log('Upload successful, showing chat interface...');
            showChatInterface();
        } else {
            hideProgress();
            showStatus(`❌ Upload failed: ${result.detail || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        hideProgress();
        showStatus(`❌ Network error: ${error.message}`, 'error');
    } finally {
        isUploading = false;
    }
}

// Progress UI
function showProgress() {
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = 'Starting upload...';
    
    // Simulate progress
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressBar.style.width = progress + '%';
        progressText.textContent = `Processing... ${Math.round(progress)}%`;
    }, 200);
    
    // Store interval ID for cleanup
    progressContainer.dataset.interval = interval;
}

function hideProgress() {
    progressContainer.style.display = 'none';
    if (progressContainer.dataset.interval) {
        clearInterval(progressContainer.dataset.interval);
    }
}

// Upload status helper
function showStatus(message, type) {
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.innerHTML = `<div class="status-message status-${type}">${message}</div>`;
    statusDiv.style.display = 'block';
}

// Chat request/response flow
function showChatInterface() {
    console.log('Showing chat interface...');
    console.log('Chat interface element:', chatInterface);
    if (chatInterface) {
        chatInterface.style.display = 'block';
        chatInterface.classList.add('fade-in');
        console.log('Chat interface should now be visible');
    } else {
        console.error('Chat interface element not found!');
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        askQuestion();
    }
}

async function askQuestion() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Add user message
    addMessage('user', query);
    queryInput.value = '';

    // Show loading
    const loadingMessage = addMessage('assistant', 'Thinking...');
    sendBtn.disabled = true;

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query })
        });

        const result = await response.json();
        loadingMessage.remove();

        if (response.ok) {
            addMessage('assistant', result.answer, result.citations);
        } else {
            addMessage('assistant', `Sorry, I encountered an error: ${result.detail || 'Unknown error'}`);
        }
    } catch (error) {
        loadingMessage.remove();
        addMessage('assistant', `Sorry, I encountered a network error: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
    }
}

function askQuickQuestion(question) {
    queryInput.value = question;
    askQuestion();
}

function addMessage(type, content, citations = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    // Convert markdown to HTML for better formatting
    let messageContent = convertMarkdownToHTML(content);
    
    if (citations && citations.length > 0) {
        console.log('Citations data:', citations);
        messageContent += `
            <div class="citations">
                <div class="citations-title">Sources</div>
                ${citations.map((c, index) => {
                    console.log(`Citation ${index}:`, c);
                    return `
                    <div class="citation-item">
                        Source ${index + 1}: ${c.filename || 'Document'} (Score: ${c.combined_score || c.relevance_score || c.score || 'N/A'})
                    </div>
                `;
                }).join('')}
            </div>
        `;
    }
    
    messageDiv.innerHTML = messageContent;
    chatMessagesContainer.appendChild(messageDiv);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    
    return messageDiv;
}

function convertMarkdownToHTML(text) {
    // Convert markdown formatting to HTML
    return text
        // Bold text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic text
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Bullet points
        .replace(/^[\s]*[-*]\s+(.+)$/gm, '<li>$1</li>')
        // Numbered lists
        .replace(/^[\s]*\d+\.\s+(.+)$/gm, '<li>$1</li>')
        // Line breaks
        .replace(/\n/g, '<br>')
        // Wrap list items in ul/ol tags
        .replace(/(<li>.*<\/li>)/gs, (match) => {
            if (match.includes('1.') || match.includes('2.') || match.includes('3.')) {
                return `<ol>${match}</ol>`;
            } else {
                return `<ul>${match}</ul>`;
            }
        });
}

// Theme toggle + persistence
function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const icon = document.querySelector('.theme-toggle');
    icon.textContent = theme === 'dark' ? 'Light' : 'Dark';
}

// UI variant toggle
function loadUIVariant() {
    const savedUI = localStorage.getItem('ui') || 'classic';
    document.documentElement.setAttribute('data-ui', savedUI);
    updateUIVariantButton(savedUI);
}

function toggleUIVariant() {
    const current = document.documentElement.getAttribute('data-ui') || 'modern';
    const next = current === 'classic' ? 'modern' : 'classic';
    document.documentElement.setAttribute('data-ui', next);
    localStorage.setItem('ui', next);
    updateUIVariantButton(next);
}

function updateUIVariantButton(variant) {
    const btn = document.querySelector('.ui-toggle');
    if (btn) btn.textContent = variant === 'classic' ? 'Modern UI' : 'Classic UI';
}

// Quick actions wiring
function updateUI() {
    // Update quick action buttons
    const quickActions = document.querySelectorAll('.quick-action');
    quickActions.forEach((action, index) => {
        const questions = [
            'What is this document about?',
            'Summarize the key points',
            'What questions can I ask about this document?'
        ];
        action.setAttribute('data-question', questions[index]);
    });
}

// Auto-show chat if backend has existing data
async function checkForExistingDocuments() {
    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: 'test' })
        });
        
        const result = await response.json();
        
        // If we get a proper response (not "No documents ingested yet"), show chat interface
        if (response.ok && !result.error) {
            console.log('Documents already loaded, showing chat interface');
            showChatInterface();
            addMessage('assistant', 'Welcome back! Your documents are already loaded. You can ask me questions about them.');
        }
    } catch (error) {
        console.log('No existing documents found');
    }
}

// Utilities
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Expose a few functions for debugging/global access
window.uploadFiles = uploadFiles;
window.askQuestion = askQuestion;
window.askQuickQuestion = askQuickQuestion;