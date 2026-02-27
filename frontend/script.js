// API base URL - use relative path to work from any host
const API_URL = '/api';

// i18n translations
const TRANSLATIONS = {
    en: {
        newChat:     'NEW CHAT',
        courses:     'Courses',
        courseCount: 'Number of courses:',
        tryAsking:   'Try asking:',
        loading:     'Loading...',
        failedLoad:  'Failed to load courses',
        placeholder: 'Ask about courses, lessons, or specific content...',
        sources:     'Sources',
        welcome:     'Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?',
        langLabel:   '中文',
    },
    zh: {
        newChat:     '新对话',
        courses:     '课程',
        courseCount: '课程数量：',
        tryAsking:   '试着问：',
        loading:     '加载中...',
        failedLoad:  '课程加载失败',
        placeholder: '询问课程、课节或具体内容...',
        sources:     '来源',
        welcome:     '欢迎使用课程资料助手！我可以帮助您解答有关课程、课节和具体内容的问题。您想了解什么？',
        langLabel:   'EN',
    },
};

let currentLang = localStorage.getItem('lang') || 'en';

function applyLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('lang', lang);
    const t = TRANSLATIONS[lang];

    // Update all data-i18n elements
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (t[key] !== undefined) el.textContent = t[key];
    });

    // Update suggested item labels and active data-question
    document.querySelectorAll('.suggested-item').forEach(btn => {
        const label = btn.getAttribute(`data-label-${lang}`);
        if (label) btn.textContent = label;
        const q = btn.getAttribute(`data-question-${lang}`) || btn.getAttribute('data-question-en');
        if (q) btn.setAttribute('data-question', q);
    });

    // Update input placeholder
    if (chatInput) chatInput.placeholder = t.placeholder;

    // Update lang toggle label
    const toggle = document.getElementById('langToggle');
    if (toggle) toggle.textContent = t.langLabel;
}

// Global state
let currentSessionId = null;

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    
    setupEventListeners();
    applyLanguage(currentLang);
    createNewSession();
    loadCourseStats();
    document.getElementById('newChatBtn').addEventListener('click', createNewSession);
    document.getElementById('langToggle').addEventListener('click', () => {
        applyLanguage(currentLang === 'en' ? 'zh' : 'en');
    });
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    
    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    if (sources && sources.length > 0) {
        const SOURCE_COLORS = [
            { rgb: '59,130,246',  color: '#93c5fd' },  // blue
            { rgb: '139,92,246',  color: '#c4b5fd' },  // purple
            { rgb: '20,184,166',  color: '#5eead4' },  // teal
            { rgb: '245,158,11',  color: '#fcd34d' },  // amber
            { rgb: '244,63,94',   color: '#fda4af' },  // rose
            { rgb: '16,185,129',  color: '#6ee7b7' },  // emerald
        ];
        const sourceLinks = sources.map((s, i) => {
            const c = SOURCE_COLORS[i % SOURCE_COLORS.length];
            const style = `--src-rgb:${c.rgb};--src-color:${c.color}`;
            if (s.url) {
                return `<a href="${s.url}" target="_blank" rel="noopener noreferrer" style="${style}">${escapeHtml(s.text)}</a>`;
            }
            return `<span style="${style};display:inline-block;padding:0.25rem 0.75rem;border-radius:999px;font-size:0.72rem;border:1px solid rgba(${c.rgb},0.35);color:${c.color}">${escapeHtml(s.text)}</span>`;
        });
        html += `
            <details class="sources-collapsible">
                <summary class="sources-header">${TRANSLATIONS[currentLang].sources}</summary>
                <div class="sources-content">${sourceLinks.join('')}</div>
            </details>
        `;
    }

    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageId;
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    // Clean up the old session on the backend
    if (currentSessionId) {
        fetch(`${API_URL}/session/${currentSessionId}`, { method: 'DELETE' }).catch(() => {});
    }
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage(TRANSLATIONS[currentLang].welcome, 'assistant', null, true);
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}