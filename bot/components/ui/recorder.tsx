import React, { useState, useRef } from "react";

interface RecorderProps {
  recordingCompleted: (transcript: string) => void;
}

const Recorder: React.FC<RecorderProps> = ({ recordingCompleted }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isRecordingComplete, setIsRecordingComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recordRef = useRef<any>(null);

  const startRecording = () => {
    setError(null); // Clear any previous errors

    if (!("webkitSpeechRecognition" in window)) {
      setError("Speech recognition is not supported in your browser.");
      return;
    }

    try {
      setIsRecording(true);
      setIsRecordingComplete(false);
      recordRef.current = new (window as any).webkitSpeechRecognition();
      recordRef.current.continuous = true;
      recordRef.current.interimResults = true;

      recordRef.current.onresult = (event: any) => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        if (event.results[0].isFinal) {
          setIsRecording(false);
          setIsRecordingComplete(true);
          recordingCompleted(transcript);
        }
      };

      recordRef.current.onerror = (event: any) => {
        console.error("Speech recognition error", event.error);
        setError(`Error: ${event.error}`);
        setIsRecording(false);
      };

      recordRef.current.start();
    } catch (err) {
      console.error("Error starting speech recognition:", err);
      setError("Error starting speech recognition");
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (recordRef.current) {
      recordRef.current.stop();
    }
    setIsRecording(false);
  };

  return (
    <div>
      {error && <p className="text-red-500">{error}</p>}
      {!isRecording && !isRecordingComplete && (
        <button
          onClick={startRecording}
          className="bg-blue-600 text-white p-2 rounded"
        >
          Start Recording
        </button>
      )}

      {isRecording && (
        <button
          onClick={stopRecording}
          className="bg-red-600 text-white p-2 rounded"
        >
          Stop Recording
        </button>
      )}

      {isRecordingComplete && <p>Recording complete!</p>}
    </div>
  );
};

export default Recorder;
