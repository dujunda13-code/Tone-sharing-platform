import { toUserFacingMessage } from "../api/client";
import { Disclosure } from "./Disclosure";

export interface SafeError {
  code: string;
  message: string;
  requestId: string;
}

/**
 * 只渲染 getApiError() 或后端 public_message 提供的自然语言文本；
 * 错误代码与 request_id 默认折叠，用户按需展开。
 */
export function ErrorPanel({ error, hint }: { error: SafeError; hint?: string }) {
  return (
    <div className="error-panel" role="alert">
      <p className="error">{toUserFacingMessage(error.message)}</p>
      <Disclosure label="技术详情">
        <div>
          错误代码：{error.code}
          {error.requestId ? ` · request_id: ${error.requestId}` : ""}
        </div>
      </Disclosure>
      {hint && <small>{hint}</small>}
    </div>
  );
}
