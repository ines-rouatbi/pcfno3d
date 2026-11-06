/** Vite can mark .gz assets as HTTP-encoded, so fetch may already have decoded them. */
export async function elevationBytes(response: Response): Promise<ArrayBuffer> {
  if(!response.ok)throw new Error(`Elevation HTTP ${response.status}`);
  const buffer=await response.arrayBuffer(),bytes=new Uint8Array(buffer);
  if(bytes[0]!==0x1f||bytes[1]!==0x8b)return buffer;
  return new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
