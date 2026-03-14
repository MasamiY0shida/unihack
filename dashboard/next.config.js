/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow the dashboard to proxy API calls to the Rust backend (port 4000)
  async rewrites() {
    return [
      {
        source:      '/api/trades/:path*',
        destination: 'http://localhost:4000/trades/:path*',
      },
      {
        source:      '/api/wallet',
        destination: 'http://localhost:4000/wallet',
      },
    ];
  },
};

module.exports = nextConfig;
