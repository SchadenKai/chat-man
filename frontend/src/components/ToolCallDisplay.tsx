import { ChevronDown, ChevronRight, Loader2, Wrench } from "lucide-react";
import { useState } from "react";
import type { ToolCall } from "../interface";

function ToolCallDisplay({ toolCall }: { toolCall: ToolCall }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="mt-2 mb-3 border border-gray-600 rounded-lg bg-gray-800 overflow-hidden">
      {/* Tool Header */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-3 py-2 flex items-center gap-2 hover:bg-gray-750 transition-colors"
      >
        <Wrench className="w-4 h-4 text-blue-400 flex-shrink-0" />
        <span className="text-sm font-medium text-gray-200 flex-1 text-left">
          {toolCall.name}
        </span>
        {toolCall.status === "running" ? (
          <Loader2 className="w-4 h-4 text-blue-400 animate-spin flex-shrink-0" />
        ) : (
          <div className="flex items-center gap-2">
            <span className="text-xs text-green-400">✓</span>
            {isExpanded ? (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-gray-400" />
            )}
          </div>
        )}
      </button>

      {/* Expandable Content */}
      {isExpanded && (
        <div className="border-t border-gray-600 bg-gray-850">
          {/* Tool Arguments */}
          {toolCall.args && (
            <div className="px-3 py-2 border-b border-gray-700">
              <div className="text-xs font-semibold text-gray-400 mb-1">
                Input:
              </div>
              <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words bg-gray-900 rounded p-2 overflow-x-auto">
                {toolCall.args}
              </pre>
            </div>
          )}

          {/* Tool Result */}
          {toolCall.result && (
            <div className="px-3 py-2">
              <div className="text-xs font-semibold text-gray-400 mb-1">
                Output:
              </div>
              <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words bg-gray-900 rounded p-2 overflow-x-auto max-h-64">
                {toolCall.result}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ToolCallDisplay;
