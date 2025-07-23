@echo off
echo Testing LLMKG MCP Server...
echo.
echo {"jsonrpc":"2.0","method":"initialize","params":{},"id":1} | target\release\llmkg_mcp_server.exe
echo.
echo Test complete.