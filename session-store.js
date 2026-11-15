// Register every terminal handler before creating the request. One promise owns
// both request errors and transaction completion, so neither can reject unseen.
function transact(database, mode, operation) {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction('sessions', mode);
    let request;
    let requestError;
    transaction.onerror = event => {
      // Request errors bubble here before transaction.error is necessarily set.
      requestError ??= event.target?.error ?? transaction.error;
    };
    transaction.onabort = () => reject(
      requestError ?? transaction.error ?? new DOMException('Transaction aborted', 'AbortError'),
    );
    transaction.oncomplete = () => resolve(request.result);
    try {
      request = operation(transaction.objectStore('sessions'));
    } catch (error) {
      // A synchronous put/getAll failure must not leave an active transaction.
      try { transaction.abort(); } catch { /* Already inactive. */ }
      reject(error);
    }
  });
}

export class SessionStore {
  constructor(indexedDBFactory = globalThis.indexedDB) {
    this.indexedDB = indexedDBFactory;
    this.databasePromise = null;
  }

  open() {
    this.databasePromise ??= new Promise((resolve, reject) => {
      const request = this.indexedDB.open('cow-counter', 1);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains('sessions')) {
          request.result.createObjectStore('sessions', {keyPath: 'id'});
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return this.databasePromise;
  }

  async write(session) {
    const snapshot = structuredClone(session);
    const database = await this.open();
    await transact(database, 'readwrite', store => store.put(snapshot));
  }

  async save(session) {
    try {
      await this.write(session);
      return {savedEvidence: true};
    } catch (error) {
      if (error?.name !== 'QuotaExceededError') throw error;
      const compact = structuredClone(session);
      compact.events = compact.events.map(event => ({...event, evidence: []}));
      await this.write(compact);
      return {savedEvidence: false};
    }
  }

  async list() {
    const database = await this.open();
    const sessions = await transact(database, 'readonly', store => store.getAll());
    return sessions.sort((a, b) => b.startedAt - a.startedAt);
  }
}
