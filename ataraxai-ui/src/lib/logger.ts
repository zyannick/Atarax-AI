
const LOG_PREFIX = "[AtaraxAI-UI]";

const logger = {
  log: (...args: any[]) => console.log(LOG_PREFIX, ...args),
  info: (...args: any[]) => console.info(`${LOG_PREFIX} [INFO]`, ...args),
  warn: (...args: any[]) => console.warn(`${LOG_PREFIX} [WARN]`, ...args),
  error: (...args: any[]) => console.error(`${LOG_PREFIX} [ERROR]`, ...args),
};

export default logger;
