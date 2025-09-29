import axios from 'axios';

// Base API configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 second timeout for complex legal queries
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth tokens if needed
apiClient.interceptors.request.use(
  (config) => {
    // Add any authentication headers here if needed
    // const token = localStorage.getItem('authToken');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling common errors
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.data);
    } else if (error.request) {
      // Request made but no response received
      console.error('Network Error:', error.request);
    } else {
      // Something else happened
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// Chat API functions
export const sendChatMessage = async (question, context = '') => {
  try {
    const response = await apiClient.post('/api/chat', {
      question: question,
      context: context || '',
    });
    
    return response;
  } catch (error) {
    // Handle specific error cases
    if (error.code === 'ECONNREFUSED') {
      throw new Error('Unable to connect to the server. Please check if the backend is running.');
    }
    throw error;
  }
};

// Function to upload documents to the dataset
export const uploadDocument = async (file) => {
  try {
    const formData = new FormData();
    formData.append('document', file);
    
    const response = await apiClient.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response;
  } catch (error) {
    throw error;
  }
};

// Function to get available legal categories
export const getLegalCategories = async () => {
  try {
    const response = await apiClient.get('/api/categories');
    return response;
  } catch (error) {
    throw error;
  }
};

// Function to get system health/status
export const getSystemHealth = async () => {
  try {
    const response = await apiClient.get('/api/health');
    return response;
  } catch (error) {
    throw error;
  }
};

export default apiClient;