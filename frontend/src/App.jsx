import { useState, useEffect, useRef } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { marked } from 'marked'
import './App.css'
import botIcon from './assets/images/bot.png'
import aiBot from './assets/images/aiBot.png'

function App() {
  const [showChat, setShowChat] = useState(true);
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const toggleChatWindow = () => {
    setShowChat(!showChat);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!message.trim()) return;
    
    try {
      setMessages(prev => [...prev, { text: message, type: 'user' }]);
      setIsLoading(true);
      setMessage('');
      
      const response = await fetch('http://127.0.0.1:5800/api', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      
      const data = await response.json();
      const formattedResponse = (
        <div 
          className="markdown-content"
          dangerouslySetInnerHTML={{ __html: marked.parse(data.data) }}
        />
      );
      
      setMessages(prev => [...prev, { text: formattedResponse, type: 'bot' }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { text: 'Error: Could not get response', type: 'bot' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <header>
        <div className="header-image container-fluid">
          <div className="row align-items-center">
            <div className="col-12 col-md-7 text-center text-md-start">
              <p className="text-white fw-bold mb-1">Welcome to our</p>
              <h1 className="text-white mb-3 display-5 fw-bold">Document Q&A Assistant</h1>
              <p className="text-white mb-4 lead">Get instant, accurate answers from your documents. Experience AI-powered assistance at your fingertips!</p>
            </div>
            <div className="col-12 col-md-5 text-center">
              <img src={aiBot} className="img-fluid w-100"  />
            </div>
          </div>
        </div>
      </header>
      <div id="chat-window-toggle" className='rounded-circle' onClick={toggleChatWindow}>
        <img className="bot-image" src={botIcon} />
        <span className={`chat-placeholder ${showChat ? 'd-none' : ''}`}>Let's chat</span>
      </div>
      <div id="chat-window" className={`${showChat ? 'chat-window-visible' : 'chat-window-hidden'}`}>
        <div id="chat-window-header" className='d-flex align-items-center justify-content-between bg-primary p-2 text-light'>
          <div id="chat-window-header-title" className='fs-4 fw-bold'>Chat</div>
          <div id="chat-window-header-close" onClick={toggleChatWindow}>
            <FontAwesomeIcon icon="fa-solid fa-xmark" size='2x' />
          </div>
        </div>
        <div id="chat-window-content">
          <div id="chat-window-content-messages">
            {messages.map((msg, index) => (
              <div 
                key={index} 
                className={`message ${msg.type === 'user' ? 'user-message' : 'bot-message'}`}
              >
                {msg.text}
              </div>
            ))}
            {isLoading && (
              <div className="message bot-message loading-dots">
                <span>.</span><span>.</span><span>.</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div id="chat-window-content-input">
            <input 
              type="text" 
              id="chat-window-content-input-input" 
              value={message}
              onChange={(e) => setMessage(e.target.value)}
            />
            <button 
              id="chat-window-content-input-send" 
              onClick={handleSendMessage}
            >
              <FontAwesomeIcon icon="fa-solid fa-paper-plane" size='2x' color='white' />
            </button>
          </div>
        </div>
      </div>
    </>
  )
}

export default App
