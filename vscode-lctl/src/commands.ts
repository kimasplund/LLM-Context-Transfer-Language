import * as vscode from 'vscode';
import { ChainTreeProvider, ChainTreeItem } from './chainProvider';

type DashboardOpener = (uri?: vscode.Uri) => void;

export function registerCommands(
    context: vscode.ExtensionContext,
    chainProvider: ChainTreeProvider,
    openDashboard: DashboardOpener
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.openDashboard', () => {
            openDashboard();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.loadChain', async (arg?: vscode.Uri | ChainTreeItem) => {
            let uri: vscode.Uri | undefined;

            if (arg instanceof vscode.Uri) {
                uri = arg;
            } else if (arg instanceof ChainTreeItem && arg.resourceUri) {
                uri = arg.resourceUri;
            } else {
                const files = await vscode.workspace.findFiles('**/*.lctl.json', '**/node_modules/**');
                if (files.length === 0) {
                    void vscode.window.showWarningMessage('No LCTL chain files found in workspace.');
                    return;
                }

                const items = files.map((file) => ({
                    label: vscode.workspace.asRelativePath(file),
                    uri: file
                }));

                const selected = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select an LCTL chain file to load'
                });

                if (selected) {
                    uri = selected.uri;
                }
            }

            if (uri) {
                openDashboard(uri);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.replay', async (arg?: string | vscode.Uri | ChainTreeItem) => {
            let chainId: string | undefined;
            let uri: vscode.Uri | undefined;

            if (typeof arg === 'string') {
                chainId = arg;
                // Try to resolve URI from Chain ID
                uri = chainProvider.findChainUri(chainId);
            } else if (arg instanceof vscode.Uri) {
                uri = arg;
            } else if (arg instanceof ChainTreeItem && arg.resourceUri) {
                uri = arg.resourceUri;
            }

            if (!uri && chainId) {
                // Try to resolve URI from Chain ID again if we only had chainId
                uri = chainProvider.findChainUri(chainId);
            }

            if (!uri) {
                // If we still don't have a URI, we can't run the CLI
                void vscode.window.showErrorMessage('Could not find chain file for replay. Please open the chain in the explorer first.');
                return;
            }

            // Create or reuse terminal
            const terminalName = 'LCTL Replay';
            let terminal = vscode.window.terminals.find(t => t.name === terminalName);
            if (!terminal) {
                terminal = vscode.window.createTerminal(terminalName);
            }

            terminal.show();
            terminal.sendText(`lctl replay "${uri.fsPath}"`);
        })
    );
}
