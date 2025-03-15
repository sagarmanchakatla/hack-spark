"use client";
import React, { useState, useEffect } from "react";
import { useChat } from "ai/react";
import Recorder from "@/components/ui/recorder";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Mic,
  MessageCircle,
  Send,
  Activity,
  Bell,
  Settings,
  User,
  Volume2,
  VolumeX,
  Globe,
} from "lucide-react";
import Link from "next/link";

interface Message {
  role: string;
  content: string;
}

const PersonalHealthAssistant: React.FC = () => {
  const [isAssistantSpeaking, setIsAssistantSpeaking] =
    useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [language, setLanguage] = useState<string>("en");
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [healthData, setHealthData] = useState<any>(null);

  const { messages, handleSubmit, setInput, input } = useChat({
    onFinish: async (message: Message) => {
      if (message.role === "assistant" && !isMuted) {
        await handleTextToVoice(message.content);
      }
    },
  });

  const handleTextToVoice = async (content: string): Promise<void> => {
    setIsAssistantSpeaking(true);
    const utterance = new SpeechSynthesisUtterance(content);
    utterance.lang = language === "en" ? "en-US" : "es-ES";

    utterance.onend = () => {
      setIsAssistantSpeaking(false);
    };

    utterance.onerror = (error) => {
      console.error("Speech synthesis error:", error);
      setIsAssistantSpeaking(false);
    };

    window.speechSynthesis.speak(utterance);
  };

  const handleRecordingComplete = (recordedInput: string) => {
    setInput(recordedInput);
    setIsRecording(false);
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
  };

  useEffect(() => {
    // Load health data when the component mounts
    const loadData = async () => {
      const response = await fetch("/api/chat");
      console.log(response);
      const data = await response.json();
      console.log(data);
      setHealthData(data);
    };
    loadData();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex h-screen">
        {/* Sidebar */}
        <div className="w-20 bg-gray-900 flex flex-col items-center py-6 space-y-6">
          <div className="text-white">
            <Activity className="h-8 w-8" />
          </div>

          <button className="p-3 rounded-xl bg-gray-800 text-white">
            <MessageCircle className="h-6 w-6" />
          </button>
          <Link href="/health-data">
            <button className="p-3 rounded-xl text-gray-400 hover:text-white">
              <Activity className="h-6 w-6" />
            </button>
          </Link>
          <button className="p-3 rounded-xl text-gray-400 hover:text-white">
            <Bell className="h-6 w-6" />
          </button>

          <div className="mt-auto flex flex-col gap-4">
            <button className="p-3 text-gray-400 hover:text-white">
              <Settings className="h-6 w-6" />
            </button>
            <button className="p-3 text-gray-400 hover:text-white">
              <User className="h-6 w-6" />
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <header className="bg-white border-b border-gray-200 p-4">
            <div className="flex justify-between items-center">
              <h1 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
                Personal Health Assistant
                <span className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                  Active
                </span>
              </h1>

              <div className="flex items-center gap-4">
                <Button
                  variant="ghost"
                  onClick={() => setIsMuted(!isMuted)}
                  className="text-gray-600"
                >
                  {isMuted ? (
                    <VolumeX className="h-5 w-5" />
                  ) : (
                    <Volume2 className="h-5 w-5" />
                  )}
                </Button>
                <Button
                  variant="ghost"
                  onClick={() =>
                    setLanguage((prev) => (prev === "en" ? "es" : "en"))
                  }
                  className="text-gray-600"
                >
                  <Globe className="h-5 w-5 mr-2" />
                  {language === "en" ? "EN" : "ES"}
                </Button>
              </div>
            </div>
          </header>

          {/* Chat Area */}
          <div className="flex-1 flex gap-4 p-4 overflow-hidden">
            <div className="flex-1 flex flex-col bg-white rounded-lg shadow-sm border border-gray-200">
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex ${
                      msg.role === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`max-w-[80%] p-4 rounded-lg ${
                        msg.role === "user"
                          ? "bg-blue-600 text-white"
                          : "bg-gray-100 text-gray-900"
                      }`}
                    >
                      <p className="leading-relaxed">{msg.content}</p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="p-4 border-t border-gray-200">
                <div className="flex items-center gap-3">
                  <Button
                    onClick={toggleRecording}
                    className={`rounded-full p-3 ${
                      isRecording ? "bg-red-500 animate-pulse" : "bg-blue-600"
                    }`}
                    disabled={isAssistantSpeaking}
                  >
                    <Mic className="h-5 w-5 text-white" />
                  </Button>

                  <form onSubmit={handleSubmit} className="flex-1 flex gap-3">
                    <input
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="Type your message..."
                      className="flex-1 px-4 py-2 rounded-lg border border-gray-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      disabled={isAssistantSpeaking}
                    />
                    <Button
                      type="submit"
                      disabled={isAssistantSpeaking}
                      className="bg-blue-600 hover:bg-blue-700"
                    >
                      <Send className="h-5 w-5" />
                    </Button>
                  </form>
                </div>

                {isRecording && (
                  <div className="mt-4">
                    <Recorder recordingCompleted={handleRecordingComplete} />
                  </div>
                )}
              </div>
            </div>

            {/* Status Card */}
            <Card className="w-72">
              <CardContent className="p-4">
                <h2 className="font-semibold mb-4">Assistant Status</h2>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Voice Output:</span>
                    <span
                      className={`text-sm font-medium ${
                        isMuted ? "text-red-600" : "text-green-600"
                      }`}
                    >
                      {isMuted ? "Muted" : "Active"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Language:</span>
                    <span className="text-sm font-medium text-blue-600">
                      {language === "en" ? "English" : "Spanish"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Status:</span>
                    <span className="text-sm font-medium text-green-600">
                      {isAssistantSpeaking ? "Speaking" : "Ready"}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PersonalHealthAssistant;
