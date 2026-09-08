// Conversations in this browser's IndexedDB, one record each; the daemon
// keeps none. A conversation:
//   { id, title, model, createdAt, updatedAt, systemPrompt, temperature,
//     maxTokens, messages: [{ role, content, reasoning?, model?, at }] }

const DB_NAME = "llmman";
const DB_VERSION = 1;
const STORE = "conversations";

let dbPromise = null;

function open() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: "id" });
        store.createIndex("updatedAt", "updatedAt");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function tx(mode, run) {
  return open().then(
    (db) =>
      new Promise((resolve, reject) => {
        const t = db.transaction(STORE, mode);
        const store = t.objectStore(STORE);
        let result;
        try {
          result = run(store);
        } catch (e) {
          reject(e);
          return;
        }
        t.oncomplete = () => resolve(result && "result" in result ? result.result : result);
        t.onerror = () => reject(t.error);
        t.onabort = () => reject(t.error);
      }),
  );
}

export function newId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** All conversations, newest first. */
export async function all() {
  const list = await tx("readonly", (store) => store.getAll());
  return (list || []).sort((a, b) => b.updatedAt - a.updatedAt);
}

export function get(id) {
  return tx("readonly", (store) => store.get(id));
}

export function put(conversation) {
  return tx("readwrite", (store) => store.put(conversation));
}

export function remove(id) {
  return tx("readwrite", (store) => store.delete(id));
}

export function clear() {
  return tx("readwrite", (store) => store.clear());
}
