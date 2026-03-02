import { useState, useEffect, useRef } from 'react';
import { Settings, Bell, MoreHorizontal, Maximize2, Minimize2, Mic, Activity, RefreshCw } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('Visual');
  const [activeNav, setActiveNav] = useState('Control Center');
  const [isLive, setIsLive] = useState(false);
  const [uptime, setUptime] = useState('00:00');
  const [entityCount, setEntityCount] = useState(0);
  const [transcript, setTranscript] = useState<{time: string, role: string, text: string}[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState<any>({});
  const [streamUrl, setStreamUrl] = useState('');
  
  // Settings State — loaded from localStorage so they survive page refresh
  const [esp32Url, setEsp32Url] = useState(() => localStorage.getItem('esp32Url') || 'http://192.168.1.10:81');
  const [mlServerUrl, setMlServerUrl] = useState(() => localStorage.getItem('mlServerUrl') || 'http://127.0.0.1:5000');
  const [geminiKey, setGeminiKey] = useState(() => localStorage.getItem('geminiKey') || '');
  const [firebaseUrl, setFirebaseUrl] = useState(() => localStorage.getItem('firebaseUrl') || 'https://pi-vision-54780-default-rtdb.asia-southeast1.firebasedatabase.app');

  // Uptime counter
  useEffect(() => {
    const start = Date.now();
    const interval = setInterval(() => {
      const diff = Math.floor((Date.now() - start) / 1000);
      const m = Math.floor(diff / 60).toString().padStart(2, '0');
      const s = (diff % 60).toString().padStart(2, '0');
      setUptime(`${m}:${s}`);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Connect to SSE
  useEffect(() => {
    const sse = new EventSource(`${mlServerUrl}/events`);
    let lastSpoken = "";
    let lastSpeakTime = 0;

    sse.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.objects) {
          setEntityCount(data.count);
          setIsLive(true);
          setPipelineStatus({
            esp32_capture: 'ok',
            python_ai_server: 'ok',
            firebase_sync: 'ok',
            detection_results: 'ok',
            dashboard: 'live'
          });

          // TTS Logic
          const now = Date.now();
          if (data.voice_message && data.voice_message !== lastSpoken && (now - lastSpeakTime > 5000)) {
             // Speak only if message changed and 5s passed, or if it's urgent
             const isUrgent = data.voice_message.startsWith("Warning") || data.voice_message.startsWith("Currency");
             
             if (isUrgent || (now - lastSpeakTime > 8000)) {
                 const utterance = new SpeechSynthesisUtterance(data.voice_message);
                 utterance.rate = 1.1;
                 window.speechSynthesis.speak(utterance);
                 lastSpoken = data.voice_message;
                 lastSpeakTime = now;
                 
                 // Update transcript
                 setTranscript(prev => [...prev.slice(-4), {
                    time: new Date().toLocaleTimeString().slice(0,8),
                    role: 'AI-CORE',
                    text: data.voice_message
                 }]);
             }
          }
        }
      } catch (err) {}
    };
    return () => sse.close();
  }, [mlServerUrl]);

  // Poll /frame every 150ms to get a live-looking feed
  useEffect(() => {
    const interval = setInterval(() => {
      setStreamUrl(`${mlServerUrl}/frame?t=${Date.now()}`);
    }, 150);
    return () => clearInterval(interval);
  }, [mlServerUrl]);

  const handleSaveSettings = async () => {
    // Persist to localStorage so settings survive page refresh
    localStorage.setItem('esp32Url', esp32Url);
    localStorage.setItem('mlServerUrl', mlServerUrl);
    localStorage.setItem('geminiKey', geminiKey);
    localStorage.setItem('firebaseUrl', firebaseUrl);

    // Tell the Python server to use the new ESP32 URL immediately (no restart needed)
    try {
      await fetch(`${mlServerUrl}/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ esp32_url: esp32Url.replace('/stream', '') }),
      });
    } catch (err) {
      console.warn('Could not reach ML server to update config:', err);
    }

    setShowSettings(false);
  };

  // Ref to keep track of the recognition instance
  const recognitionRef = useRef<any>(null);

  const [textInput, setTextInput] = useState("");

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput.trim()) return;

    const question = textInput;
    setTextInput("");

    // Update UI with user's text
    setTranscript(prev => [...prev, {
      time: new Date().toLocaleTimeString().slice(0,8),
      role: 'USER1',
      text: question
    }]);

    try {
      const res = await fetch(`${mlServerUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question })
      });
      
      const data = await res.json();
      const aiResponse = data.answer || "I didn't catch that.";

      setTranscript(prev => [...prev.slice(-4), {
        time: new Date().toLocaleTimeString().slice(0,8),
        role: 'AI-CORE',
        text: aiResponse
      }]);

      const utterance = new SpeechSynthesisUtterance(aiResponse);
      utterance.lang = 'en-US';
      utterance.rate = 1.0;
      utterance.volume = 1.0;
      
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utterance);
      console.log("Speaking (Text):", aiResponse);
      
    } catch (err) {
      console.error("Backend error:", err);
      const errUtt = new SpeechSynthesisUtterance("Connection error");
      window.speechSynthesis.speak(errUtt);
    }
  };

  const toggleMic = () => {
    // If already listening, stop it
    if (isListening) {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
        recognitionRef.current = null;
      }
      setIsListening(false);
      window.speechSynthesis.cancel();
      return;
    }

    // Start listening
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Browser does not support Speech Recognition");
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognitionRef.current = recognition;
      
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        console.log("Listening started...");
        setIsListening(true);
      };

      recognition.onresult = async (event: any) => {
        const transcriptText = event.results[0][0].transcript;
        console.log("Heard:", transcriptText);
        
        setTranscript(prev => [...prev, {
          time: new Date().toLocaleTimeString().slice(0,8),
          role: 'USER1',
          text: transcriptText
        }]);

        try {
          const res = await fetch(`${mlServerUrl}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: transcriptText })
          });
          
          const data = await res.json();
          const aiResponse = data.answer || "I didn't catch that.";

          setTranscript(prev => [...prev.slice(-4), {
            time: new Date().toLocaleTimeString().slice(0,8),
            role: 'AI-CORE',
            text: aiResponse
          }]);

          const utterance = new SpeechSynthesisUtterance(aiResponse);
          // Force English voice and settings for better compatibility
          utterance.lang = 'en-US';
          utterance.rate = 1.0;
          utterance.volume = 1.0;
          
          window.speechSynthesis.cancel(); // Clear any previous speech
          window.speechSynthesis.speak(utterance);
          console.log("Speaking:", aiResponse);
          
        } catch (err) {
          console.error("Backend error:", err);
          const errUtt = new SpeechSynthesisUtterance("Connection error");
          window.speechSynthesis.speak(errUtt);
        }
      };

      recognition.onerror = (event: any) => {
        console.error("Speech error:", event.error);
        if (event.error === 'not-allowed') {
          alert("Microphone access denied. Please allow capability.");
        }
        setIsListening(false);
      };
      
      recognition.onend = () => {
        console.log("Listening ended");
        setIsListening(false);
        recognitionRef.current = null;
      };

      recognition.start();
    } catch (e) {
      console.error("Failed to start recognition:", e);
      setIsListening(false);
    }
  };

  return (
    <>
      {/* TOP NAV */}
      <div className="h-12 bg-[rgba(10,11,15,0.95)] border-b border-[rgba(255,255,255,0.07)] flex items-center justify-between px-5 shrink-0 backdrop-blur-md z-10">
        <div className="flex items-center gap-2.5 font-display font-extrabold text-[17px] tracking-tight text-[#e8eaf0]">
          <div className="w-9 h-9 rounded-[10px] bg-gradient-to-br from-[#4f7fff] to-[#7b5ea7] flex items-center justify-center text-base">👁</div>
          <div>Vision OS <span className="font-mono text-[9px] font-medium text-[#5a5f75] border border-[rgba(255,255,255,0.12)] px-1.5 py-0.5 rounded ml-2 tracking-widest">V4.2.0</span></div>
        </div>
        <div className="flex items-center gap-1.5">
          <button className="w-8 h-8 rounded-lg border-none bg-transparent text-[#8a8fa8] hover:bg-[#111318] hover:text-[#e8eaf0] flex items-center justify-center transition-colors">
            <Bell size={16} />
          </button>
          <button 
            onClick={() => setShowSettings(true)}
            className="w-8 h-8 rounded-lg border-none bg-transparent text-[#8a8fa8] hover:bg-[#111318] hover:text-[#e8eaf0] flex items-center justify-center transition-colors"
          >
            <Settings size={16} />
          </button>
          <button className="bg-[rgba(255,107,191,0.12)] border border-[rgba(255,107,191,0.3)] text-[#ff6bbf] font-mono text-[11px] font-semibold px-3.5 py-1.5 rounded-lg cursor-pointer flex items-center gap-1.5 hover:bg-[rgba(255,107,191,0.2)] transition-colors ml-2">
            SHARE
          </button>
        </div>
      </div>

      {/* MAIN LAYOUT */}
      <div className="grid grid-cols-[220px_1fr_300px] flex-1 overflow-hidden">
        
        {/* SIDEBAR */}
        <aside className="bg-[#111318] border-r border-[rgba(255,255,255,0.07)] flex flex-col overflow-hidden">
          <div className="p-4 border-b border-[rgba(255,255,255,0.07)] flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-[10px] bg-gradient-to-br from-[#4f7fff] to-[#7b5ea7] flex items-center justify-center text-base">👁</div>
            <div>
              <div className="font-display font-bold text-sm text-[#e8eaf0]">Vision OS</div>
              <div className="text-[10px] text-[#6e9bff] font-semibold tracking-wide mt-px">V4.2.0 STABLE</div>
            </div>
          </div>

          <nav className="flex-1 p-2 flex flex-col gap-0.5">
            {['Control Center', 'Transcripts', 'Settings'].map(item => (
              <div 
                key={item}
                onClick={() => setActiveNav(item)}
                className={`flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-xs font-medium cursor-pointer transition-all tracking-wide ${activeNav === item ? 'bg-[#4f7fff] text-white' : 'text-[#8a8fa8] hover:bg-[#161820] hover:text-[#e8eaf0]'}`}
              >
                {item === 'Control Center' && <Activity size={14} />}
                {item === 'Transcripts' && <Minimize2 size={14} />} 
                {item === 'Settings' && <Settings size={14} />}
                {item}
              </div>
            ))}
            <a href="https://pi-vision-54780-default-rtdb.asia-southeast1.firebasedatabase.app" target="_blank" className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-xs font-medium cursor-pointer transition-all tracking-wide text-[#8a8fa8] hover:bg-[#161820] hover:text-[#e8eaf0]">
              <Activity size={14} /> Data
            </a>
          </nav>

          <div className="p-4 border-t border-[rgba(255,255,255,0.07)]">
            <div className="text-[9px] font-semibold tracking-[1.5px] text-[#5a5f75] mb-3">SYSTEM METRICS</div>
            <div className="mb-2.5">
              <div className="flex justify-between text-[10px] mb-1.5 text-[#8a8fa8]">
                <span>Battery</span><span className="text-[#e8eaf0] font-semibold">85%</span>
              </div>
              <div className="h-[3px] bg-[#1c1f2a] rounded-sm overflow-hidden">
                <div className="h-full rounded-sm bg-gradient-to-r from-[#4f7fff] to-[#6e9bff] w-[85%]"></div>
              </div>
            </div>
            <div className="flex justify-between items-center text-[10px] text-[#8a8fa8] mt-2">
              <span>Signal (5G)</span>
              <div className="flex items-center gap-1.5 text-[#3dffa0] font-semibold">
                <Activity size={10} /> 5G
              </div>
            </div>
            <button className="w-full mt-3 bg-gradient-to-br from-[#4f7fff] to-[#6a3de8] border-none text-white font-mono text-[11px] font-semibold p-2.5 rounded-lg cursor-pointer flex items-center justify-center gap-1.5 tracking-wide hover:opacity-90 transition-opacity">
              <RefreshCw size={12} /> System Sync
            </button>
          </div>

          <div className="p-3 px-4 border-t border-[rgba(255,255,255,0.07)] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-full bg-gradient-to-br from-[#4f7fff] to-[#ff6bbf] flex items-center justify-center text-[11px] font-bold text-white">X</div>
              <div>
                <div className="text-[11px] font-semibold text-[#e8eaf0]">user1</div>
                <div className="text-[9px] text-[#5a5f75] mt-px">Connected</div>
              </div>
            </div>
          </div>
        </aside>

        {/* CENTER */}
        <main className="flex flex-col overflow-hidden">
          <div className="h-9 border-b border-[rgba(255,255,255,0.07)] flex items-center px-4 gap-3 shrink-0 mt-2.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[#ff4f4f] shadow-[0_0_6px_#ff4f4f] animate-[pulse_1.5s_ease-in-out_infinite] shrink-0"></div>
            <span className="text-[11px] font-bold tracking-widest text-[#e8eaf0]">LIVE SESSION: AG-9902</span>
            <div className="w-px h-4 bg-[rgba(255,255,255,0.12)]"></div>
            <span className="text-[11px] text-[#8a8fa8]">📍 Location: User Current Location</span>
            <div className="ml-auto flex gap-0.5 bg-[#161820] rounded-lg p-[3px]">
              <button className="font-mono text-[10px] font-semibold px-3 py-1 rounded-md border-none cursor-pointer bg-[#4f7fff] text-white tracking-wide">Visual</button>
            </div>
            <div className="w-[30px] h-[30px] rounded-lg border border-[rgba(255,255,255,0.07)] bg-[#161820] text-[#8a8fa8] flex items-center justify-center cursor-pointer hover:text-[#e8eaf0] hover:border-[rgba(255,255,255,0.12)] ml-1">
              <Bell size={13} />
            </div>
            <div className="w-[30px] h-[30px] rounded-lg border border-[rgba(255,255,255,0.07)] bg-[#161820] text-[#8a8fa8] flex items-center justify-center cursor-pointer hover:text-[#e8eaf0] hover:border-[rgba(255,255,255,0.12)]">
              <MoreHorizontal size={13} />
            </div>
          </div>

          <div className="flex-1 p-3 px-4 flex flex-col gap-2.5 overflow-hidden">
            <div className="relative flex-1 rounded-xl overflow-hidden bg-[#080a0e] border border-[rgba(255,255,255,0.07)]">
              {/* Street Scene CSS Art / Video Feed */}
              <div className="w-full h-full relative overflow-hidden bg-gradient-to-b from-[#c8d0d8] via-[#454545] to-[#1a1a1a]">
                 {/* Placeholder Scene */}
                 {/* CSS background scene — only shown when no live feed */}
                 {!isLive && (
                   <>
                 <div className="absolute inset-0 h-1/2 bg-gradient-to-b from-[#a8b8c8] to-[#c8d4dc]"></div>
                 <div className="absolute bottom-1/2 left-0 w-[120px] h-[65%] bg-[#2e3340] border-t-2 border-[#4a5060]"></div>
                 <div className="absolute bottom-1/2 left-[110px] w-[80px] h-[55%] bg-[#353a48] border-t-2 border-[#4a5060]"></div>
                 <div className="absolute bottom-1/2 right-0 w-[130px] h-[70%] bg-[#2a2f3a] border-t-2 border-[#4a5060]"></div>
                 <div className="absolute bottom-1/2 right-[120px] w-[90px] h-[60%] bg-[#303540] border-t-2 border-[#4a5060]"></div>
                 <div className="absolute bottom-1/2 left-1/2 -translate-x-1/2 w-[60px] h-[80%] bg-[#282d38] border-t-2 border-[#4a5060]"></div>
                 <div className="absolute bottom-0 left-0 right-0 h-1/2 bg-[#2a2a2a]">
                    <div className="absolute top-[30%] left-1/2 -translate-x-1/2 w-2 h-10 bg-white/50"></div>
                    <div className="absolute top-[60px] left-1/2 -translate-x-1/2 w-2 h-10 bg-white/50"></div>
                    <div className="absolute top-[120px] left-1/2 -translate-x-1/2 w-2 h-10 bg-white/50"></div>
                 </div>
                   </>
                 )}

                 {/* Live camera feed — full cover, no blending */}
                 {isLive && streamUrl && (
                   <img
                      src={streamUrl}
                      className="absolute inset-0 w-full h-full object-cover"
                      onError={(e) => e.currentTarget.style.display = 'none'}
                   />
                 )}

                 <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    {!isLive && (
                        <div className="font-mono text-[28px] font-bold text-white/85 tracking-[6px] uppercase relative animate-[glitch_2.5s_infinite]">
                            NO SIGNAL
                        </div>
                    )}
                 </div>
              </div>

              <div className="absolute top-3 left-3 bg-[rgba(10,11,15,0.85)] border border-[rgba(255,255,255,0.12)] px-2.5 py-1 rounded-md text-[10px] font-semibold text-[#3dffa0] tracking-wide flex items-center gap-1.5 backdrop-blur-md">
                <div className="w-1.5 h-1.5 rounded-full bg-[#3dffa0] animate-[pulse-green_2s_ease-in-out_infinite]"></div>
                {isLive ? 'LIVE FEED' : 'CONNECTING...'}
              </div>

              <div className="absolute bottom-3 right-3 flex gap-1.5">
                <div className="w-8 h-8 bg-[rgba(10,11,15,0.8)] border border-[rgba(255,255,255,0.12)] rounded-lg flex items-center justify-center cursor-pointer text-[#8a8fa8] backdrop-blur-md hover:text-[#e8eaf0] hover:border-[#4f7fff]">
                    <Maximize2 size={13} />
                </div>
              </div>
            </div>

            {/* Transcript */}
            <div className={`bg-[#111318] border border-[rgba(255,255,255,0.07)] rounded-xl flex flex-col overflow-hidden shrink-0 transition-all duration-300 ${activeNav === 'Transcripts' ? 'h-[420px]' : 'h-[140px]'}`}>
              <div className="flex items-center justify-between px-3.5 py-2.5 border-b border-[rgba(255,255,255,0.07)] shrink-0">
                <div className="flex items-center gap-2 text-[10px] font-semibold tracking-widest text-[#8a8fa8]">
                  <Activity size={13} /> LIVE TRANSCRIPTION
                </div>
                <div className="text-[10px] text-[#6e9bff] cursor-pointer tracking-wide hover:text-[#e8eaf0]">EXPORT</div>
              </div>
              <div className="flex-1 overflow-y-auto p-3.5 flex flex-col gap-2">
                {transcript.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-[10px] text-[#5a5f75] tracking-wide italic">Hold mic to begin transcription...</div>
                ) : (
                    transcript.map((line, i) => (
                        <div key={i} className="grid grid-cols-[60px_80px_1fr] items-baseline gap-2.5 text-[11px] animate-[fadeUp_0.3s_ease]">
                            <span className="text-[#5a5f75] text-[10px]">{line.time}</span>
                            <span className={`text-[9px] font-bold tracking-wide px-1.5 py-0.5 rounded text-center ${line.role === 'USER1' ? 'bg-[rgba(79,127,255,0.15)] text-[#6e9bff]' : 'bg-[rgba(61,255,160,0.12)] text-[#3dffa0]'}`}>{line.role}</span>
                            <span className="text-[#e8eaf0]">"{line.text}"</span>
                        </div>
                    ))
                )}
              </div>
            </div>
          </div>
        </main>

        {/* RIGHT PANEL */}
        <aside className="bg-[#111318] border-l border-[rgba(255,255,255,0.07)] flex flex-col overflow-hidden">
          
          {/* Pipeline */}
          <div className="border-b border-[rgba(255,255,255,0.07)] p-3.5 pt-8">
            <div className="flex items-center gap-2 text-[9px] font-bold tracking-[1.5px] text-[#6e9bff] mb-3">
                <Activity size={11} /> DATA PIPELINE
            </div>
            <div className="flex items-center justify-between mb-2.5 pb-2 border-b border-[rgba(255,255,255,0.07)]">
                <span className="text-[9px] font-bold tracking-[1.5px] text-[#5a5f75]">SYSTEM STAGES</span>
                <span className="text-[9px] font-bold tracking-widest text-[#5a5f75]">{isLive ? '5/5 ACTIVE' : '0/5 ACTIVE'}</span>
            </div>
            <div className="flex flex-col gap-2.5">
                {[
                    { label: 'ESP32 Capture', key: 'esp32_capture' },
                    { label: 'Python AI Server', key: 'python_ai_server' },
                    { label: 'Firebase Sync', key: 'firebase_sync' },
                    { label: 'Detection Results', key: 'detection_results' },
                    { label: 'Dashboard', key: 'dashboard' }
                ].map((step, i) => (
                    <div key={i} className="flex items-center gap-2.5 text-xs">
                        <div className={`w-[22px] h-[22px] rounded-full border-[1.5px] bg-[#161820] flex items-center justify-center shrink-0 ${pipelineStatus[step.key] === 'ok' || pipelineStatus[step.key] === 'live' ? 'border-[#3dffa0] bg-[rgba(61,255,160,0.12)] text-[#3dffa0]' : 'border-[rgba(255,255,255,0.12)] text-[#5a5f75]'}`}>
                            {pipelineStatus[step.key] ? '✓' : '•'}
                        </div>
                        <span className="flex-1 text-[#e8eaf0] font-medium">{step.label}</span>
                        <span className={`text-[10px] font-bold tracking-wide ${pipelineStatus[step.key] === 'live' ? 'text-[#6e9bff]' : pipelineStatus[step.key] === 'ok' ? 'text-[#3dffa0]' : 'text-[#5a5f75]'}`}>
                            {pipelineStatus[step.key] === 'live' ? 'LIVE' : pipelineStatus[step.key] === 'ok' ? 'OK' : '—'}
                        </span>
                    </div>
                ))}
            </div>
          </div>

          {/* Context Insights */}
          <div className="border-b border-[rgba(255,255,255,0.07)] p-3.5 pt-8">
            <div className="flex items-center gap-2 text-[9px] font-bold tracking-[1.5px] text-[#6e9bff] mb-3">
                <Activity size={11} /> AI CONTEXT INSIGHTS
            </div>
            <div className="text-[9px] text-[#5a5f75] tracking-wide mb-1">CURRENT SCENARIO</div>
            <div className="font-display text-base font-bold text-[#e8eaf0] mb-6 leading-tight">Navigation & Commuting</div>
            <div className="bg-[#161820] border border-[rgba(255,255,255,0.12)] rounded-lg p-2.5 flex gap-2">
                <div className="w-[18px] h-[18px] shrink-0 text-[#4f7fff] mt-px"><Activity size={18} /></div>
                <div>
                    <div className="text-[10px] font-bold text-[#8a8fa8] mb-1 tracking-wide">Recommendation</div>
                    <div className="text-[11px] text-[#e8eaf0] leading-relaxed">Path clear. Proceed forward.</div>
                </div>
            </div>
          </div>

          {/* Voice Feed */}
          <div>
            <div className="flex justify-between items-center mb-2.5 px-3.5 pt-7">
                <div className="text-[9px] font-bold tracking-[1.5px] text-[#6e9bff]">ACTIVE VOICE FEED</div>
                <div className={`text-[9px] font-bold tracking-wide ${isListening ? 'text-[#3dffa0]' : 'text-[#5a5f75]'}`}>{isListening ? 'LISTENING' : 'READY'}</div>
            </div>
            <div className="flex items-center gap-2.5 px-3.5 pb-3.5 border-b border-[rgba(255,255,255,0.07)]">
                <button 
                    onClick={toggleMic}
                    className={`w-9 h-9 rounded-full border flex items-center justify-center cursor-pointer shrink-0 transition-all ${isListening ? 'bg-[rgba(61,255,160,0.2)] border-[rgba(61,255,160,0.5)] text-[#3dffa0] scale-110' : 'bg-[rgba(79,127,255,0.15)] border-[rgba(79,127,255,0.35)] text-[#4f7fff]'}`}
                >
                    <Mic size={14} />
                </button>
                <form onSubmit={handleTextSubmit} className={`flex items-center gap-[3px] h-7 flex-1 border border-[rgba(255,255,255,0.12)] rounded-lg bg-[#080a0e] px-2`}>
                   <input 
                      type="text" 
                      value={textInput}
                      onChange={(e) => setTextInput(e.target.value)}
                      placeholder='Or type here...'
                      className="w-full bg-transparent border-none outline-none text-[10px] text-[#e8eaf0] placeholder-[#5a5f75] font-mono tracking-wide"
                   />
                </form>
            </div>
          </div>

          {/* Summary */}
          <div className="p-3.5 pt-7 flex-1 flex flex-col">
            <div className="font-display text-sm font-bold text-[#e8eaf0] mb-3.5">Session Summary</div>
            <div className="grid grid-cols-2 gap-2.5 mb-3.5">
                <div className="bg-[#161820] border border-[rgba(255,255,255,0.07)] rounded-lg p-2.5">
                    <div className="text-[9px] font-bold tracking-widest text-[#5a5f75] mb-1">ENTITIES</div>
                    <div className="font-display text-[22px] font-extrabold text-[#e8eaf0] leading-none">{entityCount}</div>
                </div>
                <div className="bg-[#161820] border border-[rgba(255,255,255,0.07)] rounded-lg p-2.5">
                    <div className="text-[9px] font-bold tracking-widest text-[#5a5f75] mb-1">UPTIME</div>
                    <div className="font-display text-[22px] font-extrabold text-[#6e9bff] leading-none">{uptime}</div>
                </div>
            </div>
            <button className="w-full mt-auto bg-gradient-to-br from-[#ff4f4f] to-[#cc2020] border-none text-white font-mono text-xs font-bold p-3 rounded-[10px] cursor-pointer flex items-center justify-center gap-2 tracking-wide hover:opacity-90 hover:-translate-y-px shadow-[0_6px_20px_rgba(255,79,79,0.3)] transition-all">
                <div className="w-2 h-2 bg-white rounded-sm"></div> STOP SESSION
            </button>
          </div>

        </aside>
      </div>

      {/* SETTINGS OVERLAY */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center animate-[fadeUp_0.25s_ease]" onClick={() => setShowSettings(false)}>
            <div className="bg-[#111318] border border-[rgba(255,255,255,0.12)] rounded-2xl w-[440px] max-h-[80vh] flex flex-col shadow-[0_24px_60px_rgba(0,0,0,0.6)]" onClick={e => e.stopPropagation()}>
                <div className="flex items-center justify-between p-5 border-b border-[rgba(255,255,255,0.07)]">
                    <div className="flex items-center gap-2 text-[10px] font-bold tracking-[1.5px] text-[#6e9bff]">
                        <Settings size={13} /> SETTINGS
                    </div>
                    <button onClick={() => setShowSettings(false)} className="w-7 h-7 rounded-md border border-[rgba(255,255,255,0.07)] bg-[#161820] text-[#8a8fa8] flex items-center justify-center hover:text-[#e8eaf0] hover:border-[rgba(255,255,255,0.12)]">
                        <Minimize2 size={14} />
                    </button>
                </div>
                <div className="overflow-y-auto p-0 py-2">
                    <div className="p-5 space-y-5">
                        <div className="space-y-2">
                            <label className="flex items-center gap-2 text-[10px] font-bold tracking-[1.5px] text-[#5a5f75] uppercase">
                                <span className="text-[#8a8fa8]">📷</span> ESP32-CAM URL
                            </label>
                            <input 
                                type="text" 
                                value={esp32Url}
                                onChange={(e) => setEsp32Url(e.target.value)}
                                className="w-full bg-[#080a0e] border border-[rgba(255,255,255,0.12)] rounded-lg px-3 py-2.5 text-xs font-mono text-[#e8eaf0] focus:border-[#4f7fff] outline-none transition-colors placeholder-[#5a5f75]"
                                placeholder="http://192.168.1.10:81/stream"
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="flex items-center gap-2 text-[10px] font-bold tracking-[1.5px] text-[#5a5f75] uppercase">
                                <span className="text-[#4f7fff]">🌐</span> LOCAL ML SERVER URL
                            </label>
                            <input 
                                type="text" 
                                value={mlServerUrl}
                                onChange={(e) => setMlServerUrl(e.target.value)}
                                className="w-full bg-[#080a0e] border border-[rgba(255,255,255,0.12)] rounded-lg px-3 py-2.5 text-xs font-mono text-[#e8eaf0] focus:border-[#4f7fff] outline-none transition-colors placeholder-[#5a5f75]"
                                placeholder="http://127.0.0.1:5000"
                            />
                            <div className="flex items-center gap-2 text-[9px] text-[#5a5f75] font-mono">
                                <span>🤖 YOLOv8 object detection — run:</span>
                                <span className="text-[#3dffa0]">python vision_os_server_local_ml.py</span>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="flex items-center gap-2 text-[10px] font-bold tracking-[1.5px] text-[#5a5f75] uppercase">
                                <span className="text-[#6e9bff]">🔵</span> GEMINI API KEY (OPTIONAL CLOUD AI)
                            </label>
                            <input 
                                type="password" 
                                value={geminiKey}
                                onChange={(e) => setGeminiKey(e.target.value)}
                                className="w-full bg-[#080a0e] border border-[rgba(255,255,255,0.12)] rounded-lg px-3 py-2.5 text-xs font-mono text-[#e8eaf0] focus:border-[#4f7fff] outline-none transition-colors placeholder-[#5a5f75]"
                                placeholder="AIzaSy..."
                            />
                            <div className="text-[9px] text-[#5a5f75] leading-relaxed">
                                Enables blazing-fast, highly accurate Gemini 2.0 Flash cloud processing for chat queries.
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="flex items-center gap-2 text-[10px] font-bold tracking-[1.5px] text-[#5a5f75] uppercase">
                                <span className="text-[#ff6bbf]">🔥</span> FIREBASE DATABASE URL (OPTIONAL)
                            </label>
                            <input 
                                type="text" 
                                value={firebaseUrl}
                                onChange={(e) => setFirebaseUrl(e.target.value)}
                                className="w-full bg-[#080a0e] border border-[rgba(255,255,255,0.12)] rounded-lg px-3 py-2.5 text-xs font-mono text-[#e8eaf0] focus:border-[#4f7fff] outline-none transition-colors placeholder-[#5a5f75]"
                                placeholder="https://your-project.firebaseio.com"
                            />
                        </div>

                        <button 
                            onClick={handleSaveSettings}
                            className="w-full bg-[#2979ff] hover:bg-[#2962ff] text-white font-bold text-xs py-3 rounded-lg flex items-center justify-center gap-2 transition-colors shadow-lg shadow-blue-900/20 mt-4"
                        >
                            <span className="text-[#ffeb3b]">⚡</span> SAVE & CONNECT
                        </button>

                        <div className="bg-[#0d1117] border border-[rgba(255,255,255,0.07)] rounded-lg p-3 mt-4">
                            <div className="text-[10px] font-bold text-[#8a8fa8] mb-2 flex items-center gap-2">
                                <span>🎮</span> LOCAL ML MODE (No API Key!):
                            </div>
                            <ul className="text-[9px] text-[#c9d6e3] space-y-1 list-disc list-inside font-mono opacity-80">
                                <li>YOLOv8 Object Detection</li>
                                <li>Advanced Rupee Currency Detection</li>
                                <li>Offline TTS Audio Feedback</li>
                                <li>Run locally: <span className="text-[#e8eaf0] bg-[#1c1f2a] px-1 rounded">python vision_os_server_local_ml.py</span></li>
                                <li>Server runs on <span className="text-[#e8eaf0]">http://localhost:5000</span></li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
      )}
    </>
  );
}
