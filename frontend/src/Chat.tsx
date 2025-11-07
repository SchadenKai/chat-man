import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2 } from "lucide-react";
import {
  HttpAgent,
  type AgentSubscriber,
  type Message as AgentMessage,
} from "@ag-ui/client";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Message {
  id: number;
  role: "user" | "bot";
  content: string;
  isStreaming?: boolean;
}

export default function AIChatWithStreaming() {
  // Initialize HttpAgent
  const agentRef = useRef<HttpAgent>(
    new HttpAgent({
      url: "http://127.0.0.1:8000/v1/chat/send-message",
      headers: {
        "Content-Type": "application/json",
      },
    })
  );

  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      role: "bot",
      content: "Hello! I'm your AI assistant. How can I help you today?",
    },
  ]);
  const [input, setInput] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentBotMsgIdRef = useRef<number | null>(null);

  const scrollToBottom = (): void => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleStreamingResponse = async (
    userMessage: string
  ): Promise<void> => {
    // Add user message
    const userMsgId = Date.now();
    setMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        role: "user",
        content: userMessage,
      },
    ]);

    // Add bot message placeholder
    const botMsgId = userMsgId + 1;
    currentBotMsgIdRef.current = botMsgId;
    setMessages((prev) => [
      ...prev,
      {
        id: botMsgId,
        role: "bot",
        content: "",
        isStreaming: true,
      },
    ]);

    setIsStreaming(true);

    // Build AG-UI compatible messages array
    const agentMessages: AgentMessage[] = messages
      .filter((msg) => msg.role !== "bot" || msg.id === 1) // Include initial bot message and all user messages
      .map((msg) => ({
        id: msg.id.toString(),
        role: msg.role === "user" ? ("user" as const) : ("assistant" as const),
        content: msg.content,
      }));

    // Add the new user message
    agentMessages.push({
      id: userMsgId.toString(),
      role: "user" as const,
      content: userMessage,
    });

    // Set messages on the agent
    agentRef.current.setMessages(agentMessages);

    // Create AG-UI subscriber to handle events
    const subscriber: AgentSubscriber = {
      onTextMessageContentEvent: ({ textMessageBuffer }) => {
        // Handle streaming text content - use textMessageBuffer for accumulated text
        setMessages((prev) =>
          prev.map((msg) => {
            if (msg.id === botMsgId) {
              return {
                ...msg,
                content: textMessageBuffer || "",
              };
            }
            return msg;
          })
        );
      },
      onRunStartedEvent: (event) => {
        console.log("Run started:", event);
      },
      onRunFinishedEvent: (event) => {
        console.log("Run finished:", event);
        // Mark streaming as complete
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === botMsgId ? { ...msg, isStreaming: false } : msg
          )
        );
        setIsStreaming(false);
      },
      onRunErrorEvent: (event) => {
        console.error("Agent error:", event);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === botMsgId
              ? {
                  ...msg,
                  content: "Sorry, I encountered an error. Please try again.",
                  isStreaming: false,
                }
              : msg
          )
        );
        setIsStreaming(false);
      },
    };

    try {
      // Run the agent with AG-UI protocol
      await agentRef.current.runAgent({}, subscriber);
    } catch (error) {
      console.error("Agent execution error:", error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === botMsgId
            ? {
                ...msg,
                content: "Sorry, I encountered an error. Please try again.",
                isStreaming: false,
              }
            : msg
        )
      );
      setIsStreaming(false);
    }
  };

  const handleSend = (): void => {
    if (!input.trim() || isStreaming) return;

    const message = input.trim();
    setInput("");

    // Use either method:
    handleStreamingResponse(message);
    // OR handleSSEStreaming(message);
  };

  const handleKeyPress = (
    e: React.KeyboardEvent<HTMLTextAreaElement>
  ): void => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4 shadow-lg">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">ArnoldAI</h1>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.role === "bot" && (
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0 mt-1">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}

              <div
                className={`max-w-[70%] rounded-2xl px-4 py-3 ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-700 text-gray-100"
                }`}
              >
                <p className="whitespace-pre-wrap break-words">
                  <Markdown remarkPlugins={[remarkGfm]} skipHtml>
                    {msg.content || (msg.isStreaming ? "..." : "")}
                  </Markdown>
                  {msg.isStreaming && (
                    <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse" />
                  )}
                </p>
              </div>

              {msg.role === "user" && (
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-green-500 to-teal-600 flex items-center justify-center flex-shrink-0 mt-1">
                  <User className="w-5 h-5 text-white" />
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-gray-700 bg-gray-800 px-4 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-3 items-end">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Type your message..."
              disabled={isStreaming}
              rows={1}
              className="flex-1 bg-gray-700 text-white placeholder-gray-400 rounded-2xl px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed max-h-32"
              style={{ minHeight: "52px" }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              className="w-12 h-12 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-2xl flex items-center justify-center transition-colors flex-shrink-0"
            >
              {isStreaming ? (
                <Loader2 className="w-5 h-5 text-white animate-spin" />
              ) : (
                <Send className="w-5 h-5 text-white" />
              )}
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2 text-center">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}
