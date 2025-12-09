/**
 * API client for the LLM Council backend.
 */

const API_BASE = 'http://localhost:8001';

// Supported file types
const SUPPORTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
const SUPPORTED_DOCUMENT_TYPES = ['application/pdf', 'text/plain', 'text/csv', 'application/json', 'text/markdown'];
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB

/**
 * Convert a File to base64 string.
 */
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      // Remove "data:image/png;base64," prefix
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = (error) => reject(error);
  });
}

/**
 * Validate a file for upload.
 */
function validateFile(file) {
  if (file.size > MAX_FILE_SIZE) {
    return { valid: false, error: `File "${file.name}" is too large. Maximum size is 20MB.` };
  }

  const isImage = SUPPORTED_IMAGE_TYPES.includes(file.type);
  const isDocument = SUPPORTED_DOCUMENT_TYPES.includes(file.type);

  if (!isImage && !isDocument) {
    return { valid: false, error: `File type "${file.type}" is not supported.` };
  }

  return { valid: true };
}

/**
 * Convert files to the attachment format expected by the backend.
 */
async function formatAttachments(files) {
  if (!files || files.length === 0) {
    return null;
  }

  const attachments = await Promise.all(
    files.map(async (file) => {
      const isImage = SUPPORTED_IMAGE_TYPES.includes(file.type);
      return {
        type: isImage ? 'image' : 'document',
        media_type: file.type,
        data: await fileToBase64(file),
        filename: file.name
      };
    })
  );

  return attachments;
}

export const api = {
  /**
   * List all conversations.
   */
  async listConversations() {
    const response = await fetch(`${API_BASE}/api/conversations`);
    if (!response.ok) {
      throw new Error('Failed to list conversations');
    }
    return response.json();
  },

  /**
   * Create a new conversation.
   */
  async createConversation() {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      throw new Error('Failed to create conversation');
    }
    return response.json();
  },

  /**
   * Get a specific conversation.
   */
  async getConversation(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}`
    );
    if (!response.ok) {
      throw new Error('Failed to get conversation');
    }
    return response.json();
  },

  /**
   * Send a message in a conversation.
   * @param {string} conversationId - The conversation ID
   * @param {string} content - The message content
   * @param {File[]} files - Optional array of files to attach
   */
  async sendMessage(conversationId, content, files = []) {
    // Format attachments if provided
    const attachments = await formatAttachments(files);

    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content,
          attachments
        }),
      }
    );
    if (!response.ok) {
      throw new Error('Failed to send message');
    }
    return response.json();
  },

  /**
   * Send a message and receive streaming updates.
   * @param {string} conversationId - The conversation ID
   * @param {string} content - The message content
   * @param {function} onEvent - Callback function for each event: (eventType, data) => void
   * @param {File[]} files - Optional array of files to attach
   * @returns {Promise<void>}
   */
  async sendMessageStream(conversationId, content, onEvent, files = []) {
    // Format attachments if provided
    const attachments = await formatAttachments(files);

    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message/stream`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content,
          attachments
        }),
      }
    );

    if (!response.ok) {
      throw new Error('Failed to send message');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data);
            onEvent(event.type, event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e);
          }
        }
      }
    }
  },
};

// Export utility functions for use in components
export { validateFile, SUPPORTED_IMAGE_TYPES, SUPPORTED_DOCUMENT_TYPES, MAX_FILE_SIZE };
