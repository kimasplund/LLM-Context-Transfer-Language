import * as vscode from 'vscode';
import { ChainTreeProvider } from './chainProvider';
import { LctlDashboardPanel } from './webviewPanel';
import { registerCommands } from './commands';

let chainProvider: ChainTreeProvider;

export function activate(context: vscode.ExtensionContext): void {
    // Create tree provider with context (handles its own file watching)
    chainProvider = new ChainTreeProvider(context);

    // Create tree view with advanced features
    const treeView = vscode.window.createTreeView('lctlExplorer', {
        treeDataProvider: chainProvider,
        showCollapseAll: true,
        canSelectMany: false
    });

    context.subscriptions.push(treeView);

    // Register all commands
    registerCommands(context, chainProvider, (uri?: vscode.Uri) => {
        LctlDashboardPanel.createOrShow(context.extensionUri, uri);
    });

    // Register refresh command
    context.subscriptions.push(
        vscode.commands.registerCommand('lctl.refresh', () => {
            chainProvider.refresh();
        })
    );

    // Status bar item for active recording
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left,
        100
    );
    statusBarItem.command = 'lctl.openDashboard';
    context.subscriptions.push(statusBarItem);

    // Update status bar based on recording state
    const updateStatusBar = async () => {
        const stateFiles = await vscode.workspace.findFiles('**/.lctl-state.json', '**/node_modules/**', 1);
        if (stateFiles.length > 0) {
            statusBarItem.text = '$(record) LCTL Recording';
            statusBarItem.tooltip = 'LCTL tracing is active. Click to open dashboard.';
            statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
            statusBarItem.show();
        } else {
            statusBarItem.hide();
        }
    };

    // Initial check and watch for state changes
    updateStatusBar();
    const stateWatcher = vscode.workspace.createFileSystemWatcher('**/.lctl-state.json');
    stateWatcher.onDidCreate(updateStatusBar);
    stateWatcher.onDidDelete(updateStatusBar);
    context.subscriptions.push(stateWatcher);
}

export function deactivate(): void {
    chainProvider?.dispose();
}
